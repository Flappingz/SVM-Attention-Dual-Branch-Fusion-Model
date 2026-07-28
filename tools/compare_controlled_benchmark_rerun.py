"""Compare neural outputs between two controlled-benchmark executions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_reruns(first_dir: Path, second_dir: Path) -> dict[str, Any]:
    first_dir = first_dir.resolve()
    second_dir = second_dir.resolve()
    first_root = first_dir.parents[1]
    second_root = second_dir.parents[1]
    if first_root != second_root:
        raise ValueError("Both benchmarks must belong to the same artifact root")
    first = _read_json(first_dir / "summary.json")
    second = _read_json(second_dir / "summary.json")
    results: dict[str, list[dict[str, Any]]] = {}
    all_exact = True
    for model_id in ("qwen_vl_mean_mlp", "hierarchical_attention_full"):
        first_rows = first["model_results"][model_id]["per_repetition"]
        second_rows = second["model_results"][model_id]["per_repetition"]
        if len(first_rows) != len(second_rows):
            raise ValueError(f"{model_id} repetition counts differ")
        rows: list[dict[str, Any]] = []
        for original, rerun in zip(first_rows, second_rows, strict=True):
            if (
                original["split_seed"] != rerun["split_seed"]
                or original["split_id"] != rerun["split_id"]
            ):
                raise ValueError(f"{model_id} rerun does not use identical splits")
            original_dir = first_root / "runs" / original["run_id"]
            rerun_dir = first_root / "runs" / rerun["run_id"]
            oof_exact = _sha256(original_dir / "oof_predictions.jsonl") == _sha256(
                rerun_dir / "oof_predictions.jsonl"
            )
            original_checkpoints = sorted(
                original_dir.glob("fold-*/best_model.safetensors")
            )
            rerun_checkpoints = sorted(
                rerun_dir.glob("fold-*/best_model.safetensors")
            )
            checkpoints_exact = len(original_checkpoints) == len(
                rerun_checkpoints
            ) and all(
                _sha256(left) == _sha256(right)
                for left, right in zip(
                    original_checkpoints, rerun_checkpoints, strict=True
                )
            )
            metrics_exact = original["oof_metrics"] == rerun["oof_metrics"]
            exact = oof_exact and checkpoints_exact and metrics_exact
            all_exact = all_exact and exact
            rows.append(
                {
                    "split_seed": original["split_seed"],
                    "metrics_exact": metrics_exact,
                    "oof_predictions_byte_exact": oof_exact,
                    "checkpoint_count": len(original_checkpoints),
                    "checkpoints_byte_exact": checkpoints_exact,
                    "exact": exact,
                }
            )
        results[model_id] = rows
    return {
        "verdict": "REPRODUCIBLE" if all_exact else "MISMATCH",
        "comparison": "exact deterministic neural rerun",
        "first_benchmark_id": first["benchmark_id"],
        "second_benchmark_id": second["benchmark_id"],
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-benchmark-dir", type=Path, required=True)
    parser.add_argument("--second-benchmark-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_reruns(
                args.first_benchmark_dir, args.second_benchmark_dir
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
