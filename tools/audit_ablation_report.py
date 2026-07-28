"""Independently audit the modality and architecture ablation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.experiments.ablation_report import METRICS
from ocd_v3.features.training_data import PreparedFeatureSet, fit_metadata_standardizer
from ocd_v3.io import read_jsonl, write_json_atomic


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context}: {actual} != {expected}")


def _oof_fingerprint(rows: list[dict[str, Any]]) -> str:
    canonical = sorted(
        (str(row["subject_id"]), int(row["label"]), int(row["outer_fold"]))
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _expected_condition(family: str, condition: str) -> dict[str, Any]:
    if family == "modality":
        mapping = {
            "text": (["text"], "excluded", "full"),
            "image": (["image"], "excluded", "full"),
            "metadata": (None, "only", None),
            "text+image": (["text", "image"], "excluded", "full"),
            "text+metadata": (["text"], "included", "full"),
            "text+image+metadata": (["text", "image"], "included", "full"),
        }
        modalities, metadata, architecture = mapping[condition]
        return {
            "embedding_input_modalities": modalities,
            "metadata_condition": metadata,
            "architecture_variant": architecture,
        }
    return {
        "embedding_input_modalities": ["text", "image"],
        "metadata_condition": "included",
        "architecture_variant": condition,
    }


def _audit_scaler(
    *,
    feature_set: PreparedFeatureSet,
    split: dict[str, Any],
    fold: int,
    scaler_path: Path,
) -> None:
    training_ids = [
        str(row["subject_id"])
        for row in split["assignments"]
        if int(row["outer_fold"]) == fold and str(row["role"]) == "train"
    ]
    expected = fit_metadata_standardizer(feature_set, training_ids)
    observed = _read_json(scaler_path)
    np.testing.assert_allclose(observed["mean"], expected.mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(observed["scale"], expected.scale, rtol=0.0, atol=1e-12)


def _audit_neural_run(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    artifact_root: Path,
    expected: dict[str, Any],
) -> tuple[int, int]:
    parameters = manifest["parameters"]
    config = parameters["full_experiment"]
    model_config = config["model"]
    architecture = model_config.get("architecture_variant", "full")
    if architecture != expected["architecture_variant"]:
        raise ValueError("Neural run architecture differs from the matrix condition")
    if parameters["metadata_condition"] != expected["metadata_condition"]:
        raise ValueError("Neural run metadata condition differs from the matrix condition")
    if parameters.get("embedding_input_modalities", ["text", "image"]) != expected[
        "embedding_input_modalities"
    ]:
        raise ValueError("Neural run embedding modalities differ from the matrix condition")

    feature_set = PreparedFeatureSet(
        artifact_root / "features" / str(parameters["feature_id"])
    )
    split = _read_json(
        artifact_root / "splits" / str(summary["split_id"]) / "assignments.json"
    )
    checkpoint_count = 0
    tensor_count = 0
    for fold in range(5):
        fold_dir = run_dir / f"fold-{fold}"
        tensors = load_file(fold_dir / "best_model.safetensors")
        if not tensors or any(not np.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Neural checkpoint contains no tensors or non-finite values")
        names = set(tensors)
        has_post_attention = any(name.startswith("post_encoder.attention.") for name in names)
        has_user_attention = any(name.startswith("user_encoder.attention.") for name in names)
        has_post_ffn = any(name.startswith("post_encoder.feed_forward.") for name in names)
        has_user_ffn = any(name.startswith("user_encoder.feed_forward.") for name in names)
        expected_blocks = {
            "full": (True, True, True, True),
            "without_post_cross_attention": (False, True, True, True),
            "without_user_self_attention": (True, False, True, True),
            "mean_pooling_at_both_levels": (False, False, False, False),
        }[architecture]
        if (has_post_attention, has_user_attention, has_post_ffn, has_user_ffn) != (
            expected_blocks
        ):
            raise ValueError("Checkpoint block schema differs from its architecture label")
        has_metadata_parameters = any(
            name.startswith("metadata_projection.") for name in names
        )
        if has_metadata_parameters != (expected["metadata_condition"] == "included"):
            raise ValueError("Checkpoint metadata parameters differ from the input label")
        scaler_path = fold_dir / "metadata_standardizer.json"
        if expected["metadata_condition"] == "included":
            _audit_scaler(
                feature_set=feature_set,
                split=split,
                fold=fold,
                scaler_path=scaler_path,
            )
        elif scaler_path.exists():
            raise ValueError("No-metadata run unexpectedly saved a metadata scaler")
        checkpoint_count += 1
        tensor_count += len(tensors)
    return checkpoint_count, tensor_count


def _audit_metadata_run(
    *, run_dir: Path, summary: dict[str, Any], artifact_root: Path
) -> tuple[int, int]:
    feature_set = PreparedFeatureSet(
        artifact_root / "features" / str(summary["feature_id"])
    )
    split = _read_json(
        artifact_root / "splits" / str(summary["split_id"]) / "assignments.json"
    )
    model_count = 0
    value_count = 0
    for fold in range(5):
        fold_dir = run_dir / f"fold-{fold}"
        _audit_scaler(
            feature_set=feature_set,
            split=split,
            fold=fold,
            scaler_path=fold_dir / "metadata_standardizer.json",
        )
        with np.load(fold_dir / "model_parameters.npz", allow_pickle=False) as payload:
            values = [payload[name] for name in payload.files if name != "schema_version"]
            if any(not np.isfinite(value).all() for value in values):
                raise ValueError("Metadata checkpoint contains non-finite values")
            value_count += sum(value.size for value in values)
        model_count += 1
    return model_count, value_count


def _audit_repeated_summary(
    *,
    summary_path: Path,
    artifact_root: Path,
    expected: dict[str, Any],
    reference_fingerprints: dict[int, str],
) -> dict[str, Any]:
    repeated = _read_json(summary_path)
    metric_values = {metric: [] for metric in METRICS}
    prediction_rows = 0
    checkpoint_count = 0
    checkpoint_value_count = 0
    for repetition in repeated["per_repetition"]:
        run_dir = artifact_root / "runs" / str(repetition["run_id"])
        summary = _read_json(run_dir / "summary.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        rows = read_jsonl(run_dir / "oof_predictions.jsonl")
        if len(rows) != 177 or len({str(row["subject_id"]) for row in rows}) != 177:
            raise ValueError("Ablation run does not contain 177 unique OOF subjects")
        seed = int(repetition["split_seed"])
        fingerprint = _oof_fingerprint(rows)
        reference = reference_fingerprints.setdefault(seed, fingerprint)
        if fingerprint != reference:
            raise ValueError("Ablation OOF subjects, labels, or folds are not aligned")
        metrics = binary_metrics(
            [int(row["label"]) for row in rows],
            [float(row["score"]) for row in rows],
            float(summary["threshold"]),
        ).to_dict()
        for metric in METRICS:
            _assert_close(
                float(metrics[metric]),
                float(repetition["oof_metrics"][metric]),
                f"{repetition['run_id']} {metric}",
            )
            metric_values[metric].append(float(metrics[metric]))
        if expected["metadata_condition"] == "only":
            models, values = _audit_metadata_run(
                run_dir=run_dir, summary=summary, artifact_root=artifact_root
            )
        else:
            models, values = _audit_neural_run(
                run_dir=run_dir,
                manifest=manifest,
                summary=summary,
                artifact_root=artifact_root,
                expected=expected,
            )
        checkpoint_count += models
        checkpoint_value_count += values
        prediction_rows += len(rows)
    for metric, values in metric_values.items():
        recomputed = asdict(mean_sd_ci95(values))
        for field, value in recomputed.items():
            _assert_close(
                float(value),
                float(repeated["metric_summary"][metric][field]),
                f"repeated {metric} {field}",
            )
    return {
        "repeated_cv_id": repeated["repeated_cv_id"],
        "runs": len(repeated["per_repetition"]),
        "oof_prediction_rows": prediction_rows,
        "fold_checkpoints": checkpoint_count,
        "checkpoint_numeric_values_or_tensors": checkpoint_value_count,
        "metrics_recomputed": list(METRICS),
    }


def _audit_feature_conditions(report: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    feature_conditions: dict[str, PreparedFeatureSet] = {}
    cohort_ids: set[str] | None = None
    for condition, path in report["modality"]["summary_paths"].items():
        if condition == "metadata":
            continue
        repeated = _read_json(Path(path))
        feature_id = str(repeated["feature_id"])
        feature_set = PreparedFeatureSet(artifact_root / "features" / feature_id)
        expected = _expected_condition("modality", condition)[
            "embedding_input_modalities"
        ]
        observed = feature_set.manifest.get(
            "embedding_input_modalities", ["text", "image"]
        )
        if observed != expected:
            raise ValueError("Feature manifest differs from its modality condition")
        if condition in {"text", "image"}:
            if feature_set.manifest.get("embedding_derivation") is not None:
                raise ValueError("Single-modality feature set reused joint embeddings")
            if int(
                feature_set.manifest.get("build_runtime", {}).get(
                    "embedded_posts_this_invocation", 0
                )
            ) < 1:
                raise ValueError("Single-modality feature set has no re-encoding record")
        feature_conditions[condition] = feature_set
        first_run = _read_json(Path(path))["per_repetition"][0]
        rows = read_jsonl(
            artifact_root / "runs" / str(first_run["run_id"]) / "oof_predictions.jsonl"
        )
        ids = {str(row["subject_id"]) for row in rows}
        cohort_ids = ids if cohort_ids is None else cohort_ids
        if ids != cohort_ids:
            raise ValueError("Feature-condition cohorts differ")
    assert cohort_ids is not None
    reference = feature_conditions["text+image+metadata"]
    bundles_checked = 0
    for subject_id in sorted(cohort_ids):
        reference_payload = reference.load(subject_id)
        reference_posts = reference_payload["post_ids"].tolist()
        for condition, feature_set in feature_conditions.items():
            payload = feature_set.load(subject_id)
            if payload["post_ids"].tolist() != reference_posts:
                raise ValueError("Modality feature post IDs or order differ")
            if not np.isfinite(payload["content_embeddings"]).all():
                raise ValueError("Modality feature contains non-finite embeddings")
            presence = payload["modality_presence"]
            if condition.startswith("text") and "image" not in condition:
                if presence[:, 1:].any():
                    raise ValueError("Text-only feature bundle contains visual inputs")
            if condition == "image" and presence[:, 0].any():
                raise ValueError("Image-only feature bundle contains text inputs")
            bundles_checked += 1
    return {
        "feature_sets": len(feature_conditions),
        "subjects": len(cohort_ids),
        "bundles_checked": bundles_checked,
        "post_ids_and_order_aligned": True,
        "excluded_modality_presence_zero": True,
        "embeddings_finite": True,
    }


def audit(*, report_path: Path, artifact_root: Path, output_path: Path) -> dict[str, Any]:
    report = _read_json(report_path.resolve())
    artifact_root = artifact_root.resolve()
    feature_audit = _audit_feature_conditions(report, artifact_root)
    reference_fingerprints: dict[int, str] = {}
    summaries: dict[str, Any] = {}
    audited_paths: set[Path] = set()
    for family in ("modality", "architecture"):
        for condition, value in report[family]["summary_paths"].items():
            path = Path(value).resolve()
            if path in audited_paths:
                continue
            audited_paths.add(path)
            summaries[f"{family}:{condition}"] = _audit_repeated_summary(
                summary_path=path,
                artifact_root=artifact_root,
                expected=_expected_condition(family, condition),
                reference_fingerprints=reference_fingerprints,
            )
    result = {
        "schema_version": 1,
        "verification_status": "VERIFIED",
        "report_path": str(report_path.resolve()),
        "feature_audit": feature_audit,
        "summary_audits": summaries,
        "shared_oof_fingerprints": dict(sorted(reference_fingerprints.items())),
        "interpretation_boundary": report["interpretation_boundary"],
    }
    write_json_atomic(output_path.resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        report_path=Path(args.report),
        artifact_root=Path(args.artifact_root),
        output_path=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
