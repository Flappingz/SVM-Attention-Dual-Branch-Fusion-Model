"""Independently recompute and verify a completed controlled benchmark artifact."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.experiments.controlled_benchmark import (
    ControlledSplitRecord,
    audit_controlled_inputs,
    audit_model_oof_alignment,
    matched_model_comparisons,
)
from ocd_v3.experiments.controlled_benchmark_config import (
    CONTROLLED_METRICS,
    load_controlled_benchmark_config,
)
from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import read_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch for {name}: {actual} != {expected}")


def audit_completed_benchmark(
    *,
    benchmark_dir: Path,
    dataset_dir: Path,
    qwen_feature_dir: Path,
    chinese_transformer_feature_dir: Path,
    study_config_path: Path,
    benchmark_config_path: Path,
) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    artifact_root = benchmark_dir.parents[1]
    summary = _read_json(benchmark_dir / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("Controlled benchmark is not complete")
    study = load_config(study_config_path)
    protocol = load_controlled_benchmark_config(benchmark_config_path)
    if study.artifact_root != artifact_root:
        raise ValueError("Benchmark directory is outside the configured artifact root")
    if summary.get("protocol") != protocol.public_dict():
        raise ValueError("Stored protocol differs from the frozen benchmark config")
    deterministic_runtime = summary.get("deterministic_runtime")
    if not isinstance(deterministic_runtime, dict) or deterministic_runtime.get(
        "cublas_workspace_config"
    ) != ":4096:8":
        raise ValueError("Benchmark does not freeze the deterministic cuBLAS runtime")

    dataset = PreparedDataset(dataset_dir)
    qwen_features = PreparedFeatureSet(qwen_feature_dir)
    chinese_features = PreparedChineseTransformerFeatureSet(
        chinese_transformer_feature_dir
    )
    input_audit = audit_controlled_inputs(
        dataset=dataset,
        study_config=study,
        benchmark_config=protocol,
        qwen_feature_set=qwen_features,
        chinese_feature_set=chinese_features,
    )
    stored_input_audit = summary["material_passport"]["data"]["cohort"]
    if input_audit != stored_input_audit:
        raise ValueError("Recomputed cohort/input audit differs from the benchmark")

    split_records: list[ControlledSplitRecord] = []
    split_ids = summary["split_ids"]
    if len(split_ids) != 8 or len(set(split_ids)) != 8:
        raise ValueError("Benchmark does not contain eight unique shared splits")
    for seed, split_id in zip(protocol.splits.seeds, split_ids, strict=True):
        split_dir = artifact_root / "splits" / str(split_id)
        split_summary = _read_json(split_dir / "summary.json")
        assignments_path = split_dir / "assignments.json"
        assignments = _read_json(assignments_path)
        if int(assignments["seed"]) != seed:
            raise ValueError("Split seed order differs from the frozen protocol")
        if assignments.get("minimum_available_posts_per_subject") != 21:
            raise ValueError("Split cohort rule is not strict available_post_count > 20")
        split_records.append(
            ControlledSplitRecord(seed, assignments_path, split_summary)
        )

    runs_by_model: dict[str, list[SimpleNamespace]] = {}
    recomputed_model_summaries: dict[str, dict[str, Any]] = {}
    recomputed_prediction_rows = 0
    for model_id in protocol.models:
        model_result = summary["model_results"][model_id]
        repetitions = model_result["per_repetition"]
        if len(repetitions) != 8:
            raise ValueError(f"{model_id} does not contain eight repetitions")
        runs: list[SimpleNamespace] = []
        recomputed_repetitions: list[dict[str, Any]] = []
        for seed, split_id, repetition in zip(
            protocol.splits.seeds, split_ids, repetitions, strict=True
        ):
            if (
                int(repetition["split_seed"]) != seed
                or repetition["split_id"] != split_id
            ):
                raise ValueError(f"{model_id} repetition is not matched to shared splits")
            run_dir = artifact_root / "runs" / str(repetition["run_id"])
            manifest = _read_json(run_dir / "run_manifest.json")
            run_summary = _read_json(run_dir / "summary.json")
            if manifest.get("dataset_id") != dataset.dataset_id:
                raise ValueError(f"{model_id} run uses a different dataset")
            if manifest.get("split_id") != split_id:
                raise ValueError(f"{model_id} run uses a different split")
            parameters = manifest.get("parameters", {})
            if parameters.get("keyword_condition") != "removed":
                raise ValueError(f"{model_id} is not Keyword-post-removed")
            cohort = parameters.get("cohort_eligibility")
            if cohort is not None and cohort.get(
                "minimum_available_posts_per_subject"
            ) != 21:
                raise ValueError(f"{model_id} run uses a different cohort threshold")
            if model_id in {
                "qwen_vl_mean_mlp",
                "hierarchical_attention_full",
            }:
                if manifest.get("environment", {}).get(
                    "cublas_workspace_config"
                ) != ":4096:8":
                    raise ValueError(f"{model_id} lacks deterministic cuBLAS provenance")
                context = parameters.get("run_context", {})
                if context.get("deterministic_runtime") != deterministic_runtime:
                    raise ValueError(f"{model_id} runtime identity differs from benchmark")

            rows = read_jsonl(run_dir / "oof_predictions.jsonl")
            metrics = binary_metrics(
                [int(row["label"]) for row in rows],
                [float(row["score"]) for row in rows],
                threshold=protocol.classification_threshold,
            ).to_dict()
            for metric in CONTROLLED_METRICS:
                _assert_close(
                    float(metrics[metric]),
                    float(run_summary["oof_metrics"][metric]),
                    f"{model_id}/{seed}/{metric}",
                )
                _assert_close(
                    float(metrics[metric]),
                    float(repetition["oof_metrics"][metric]),
                    f"{model_id}/{seed}/repeated/{metric}",
                )
            recomputed_repetitions.append(
                {
                    **repetition,
                    "oof_metrics": {
                        **repetition["oof_metrics"],
                        **{metric: metrics[metric] for metric in CONTROLLED_METRICS},
                    },
                }
            )
            recomputed_prediction_rows += len(rows)
            runs.append(SimpleNamespace(run_dir=run_dir))
        for metric in CONTROLLED_METRICS:
            interval = asdict(
                mean_sd_ci95(
                    float(row["oof_metrics"][metric])
                    for row in recomputed_repetitions
                )
            )
            for key, value in interval.items():
                if key == "n":
                    if value != model_result["metric_summary"][metric][key]:
                        raise ValueError(f"Repeated summary count mismatch for {model_id}")
                else:
                    _assert_close(
                        float(value),
                        float(model_result["metric_summary"][metric][key]),
                        f"{model_id}/aggregate/{metric}/{key}",
                    )
        recomputed_model_summaries[model_id] = {
            "per_repetition": recomputed_repetitions
        }
        runs_by_model[model_id] = runs

    alignment = audit_model_oof_alignment(
        split_records=split_records, runs_by_model=runs_by_model
    )
    if alignment != summary["alignment_audit"]:
        raise ValueError("Independent OOF alignment audit differs from stored audit")
    comparisons = matched_model_comparisons(recomputed_model_summaries)
    if comparisons != summary["paired_comparisons"]:
        raise ValueError("Independent paired comparisons differ from stored comparisons")
    if summary.get("trained_fold_models") != 160:
        raise ValueError("Controlled benchmark must contain 160 trained fold models")

    return {
        "status": "VERIFIED",
        "benchmark_id": summary["benchmark_id"],
        "dataset_id": dataset.dataset_id,
        "subjects": input_audit["subjects_included"],
        "models": 4,
        "split_repetitions": 8,
        "outer_folds": 5,
        "trained_fold_models": 160,
        "oof_prediction_rows_recomputed": recomputed_prediction_rows,
        "keyword_condition": "removed",
        "cohort_rule": "available_post_count > 20",
        "checks": {
            "feature_post_ids_and_order": True,
            "shared_splits_subjects_labels_and_folds": True,
            "run_level_oof_metrics": True,
            "repeated_metric_intervals": True,
            "matched_pairwise_comparisons": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--qwen-feature-dir", type=Path, required=True)
    parser.add_argument("--chinese-transformer-feature-dir", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_completed_benchmark(
                benchmark_dir=args.benchmark_dir,
                dataset_dir=args.dataset_dir,
                qwen_feature_dir=args.qwen_feature_dir,
                chinese_transformer_feature_dir=(
                    args.chinese_transformer_feature_dir
                ),
                study_config_path=args.study_config,
                benchmark_config_path=args.benchmark_config,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
