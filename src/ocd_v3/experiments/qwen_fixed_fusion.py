"""Auditable fixed fusion of aligned user-level OOF endpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.experiments.repeated_oof import METRIC_NAMES
from ocd_v3.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from ocd_v3.provenance import git_state


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _indexed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["subject_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate OOF subject: {path}")
    return indexed


def build_fixed_qwen_sequence_fusion(
    *,
    artifact_root: Path,
    primary_summary_path: Path,
    secondary_summary_path: Path,
    output_root: Path,
    repository: Path,
    secondary_weight: float,
    selection_provenance: str,
    threshold: float = 0.5,
    secondary_kind: str = "qwen_hierarchical",
) -> tuple[Path, dict[str, Any]]:
    if not 0.0 <= secondary_weight <= 1.0:
        raise ValueError("secondary_weight must be in [0, 1]")
    if not selection_provenance.strip():
        raise ValueError("selection_provenance must be explicit")
    if secondary_kind not in {"qwen_hierarchical", "metadata"}:
        raise ValueError("secondary_kind must be qwen_hierarchical or metadata")
    artifact_root = artifact_root.resolve()
    primary = _read_json(primary_summary_path.resolve())
    secondary = _read_json(secondary_summary_path.resolve())
    if primary.get("dataset_id") != secondary.get("dataset_id"):
        raise ValueError("Qwen endpoints use different datasets")
    if primary.get("split_ids") != secondary.get("split_ids"):
        raise ValueError("Qwen endpoints use different split IDs")
    primary_parameters = primary.get("parameters", {})
    keyword_condition = primary_parameters.get("keyword_condition")
    if keyword_condition not in {"original", "removed"}:
        raise ValueError("Primary endpoint has an unsupported keyword condition")
    if (
        primary_parameters.get("configuration", {})
        .get("encoder", {})
        .get("model_id")
        != "Qwen/Qwen3-VL-Embedding-2B"
    ):
        raise ValueError("Primary endpoint is not Qwen3-VL-Embedding-2B")
    secondary_first_run = secondary["per_repetition"][0]
    secondary_run_manifest = _read_json(
        artifact_root
        / "runs"
        / str(secondary_first_run["run_id"])
        / "run_manifest.json"
    )
    secondary_manifest_parameters = secondary_run_manifest.get("parameters", {})
    if secondary_kind == "qwen_hierarchical":
        secondary_parameters = secondary_manifest_parameters.get("full_experiment", {})
        if secondary.get("keyword_condition") != keyword_condition:
            raise ValueError("Qwen endpoints use different keyword conditions")
        if (
            secondary_parameters.get("encoder", {}).get("model_id")
            != "Qwen/Qwen3-VL-Embedding-2B"
        ):
            raise ValueError("Secondary endpoint is not Qwen3-VL-Embedding-2B")
    else:
        if (
            secondary_run_manifest.get("experiment_id")
            != "metadata_only_mean_logistic_regression"
            or secondary_manifest_parameters.get("baseline")
            != "metadata_only_logistic_regression"
        ):
            raise ValueError("Secondary endpoint is not the metadata-only expert")
        if secondary_manifest_parameters.get("keyword_condition") != keyword_condition:
            raise ValueError("Qwen and metadata endpoints use different keyword conditions")

    secondary_by_seed = {
        int(row["split_seed"]): row for row in secondary["per_repetition"]
    }
    fused_rows: list[dict[str, Any]] = []
    per_repetition: list[dict[str, Any]] = []
    reference_subjects: dict[str, int] | None = None
    for primary_repetition in primary["per_repetition"]:
        seed = int(primary_repetition["split_seed"])
        secondary_repetition = secondary_by_seed[seed]
        if primary_repetition["split_id"] != secondary_repetition["split_id"]:
            raise ValueError("Paired Qwen endpoints use different splits")
        primary_run_id = str(primary_repetition["run_id"])
        secondary_run_id = str(secondary_repetition["run_id"])
        primary_oof = _indexed(
            artifact_root / "runs" / primary_run_id / "oof_predictions.jsonl"
        )
        secondary_oof = _indexed(
            artifact_root / "runs" / secondary_run_id / "oof_predictions.jsonl"
        )
        if set(primary_oof) != set(secondary_oof):
            raise ValueError("Paired Qwen endpoints cover different users")
        subjects = {
            subject_id: int(row["label"])
            for subject_id, row in primary_oof.items()
        }
        if reference_subjects is None:
            reference_subjects = subjects
        elif subjects != reference_subjects:
            raise ValueError("Subject labels differ between repetitions")
        labels: list[int] = []
        scores: list[float] = []
        for subject_id in sorted(primary_oof):
            first = primary_oof[subject_id]
            second = secondary_oof[subject_id]
            if (
                int(first["label"]) != int(second["label"])
                or int(first["outer_fold"]) != int(second["outer_fold"])
            ):
                raise ValueError("Paired Qwen labels/folds differ")
            score = (1.0 - secondary_weight) * float(
                first["score"]
            ) + secondary_weight * float(second["score"])
            label = int(first["label"])
            labels.append(label)
            scores.append(score)
            fused_rows.append(
                {
                    "split_seed": seed,
                    "split_id": str(primary_repetition["split_id"]),
                    "outer_fold": int(first["outer_fold"]),
                    "subject_id": subject_id,
                    "label": label,
                    "primary_score": float(first["score"]),
                    "secondary_score": float(second["score"]),
                    "primary_weight": 1.0 - secondary_weight,
                    "secondary_weight": secondary_weight,
                    "score": score,
                    "threshold": threshold,
                    "prediction": int(score >= threshold),
                }
            )
        per_repetition.append(
            {
                "split_seed": seed,
                "split_id": str(primary_repetition["split_id"]),
                "primary_run_id": primary_run_id,
                "secondary_run_id": secondary_run_id,
                "oof_metrics": binary_metrics(labels, scores, threshold).to_dict(),
            }
        )

    source_git = git_state(repository.resolve())
    experiment_id = (
        "fixed_pure_qwen_sequence_score_fusion"
        if secondary_kind == "qwen_hierarchical"
        else "fixed_qwen_sequence_metadata_score_fusion"
    )
    identity = {
        "experiment_id": experiment_id,
        "dataset_id": primary["dataset_id"],
        "feature_id": primary_parameters["feature_id"],
        "keyword_condition": keyword_condition,
        "primary_repeated_cv_id": primary["repeated_cv_id"],
        "secondary_repeated_cv_id": secondary["repeated_cv_id"],
        "split_seeds": [int(row["split_seed"]) for row in per_repetition],
        "split_ids": [str(row["split_id"]) for row in per_repetition],
        "primary_weight": 1.0 - secondary_weight,
        "secondary_weight": secondary_weight,
        "threshold": threshold,
        "selection_provenance": selection_provenance,
        "source_code": {
            "commit": source_git["commit"],
            "dirty": source_git["dirty"],
            "worktree_sha256": source_git["worktree_sha256"],
        },
    }
    if secondary_kind == "metadata":
        identity.update(
            {
                "primary_feature_id": primary_parameters["feature_id"],
                "secondary_feature_id": secondary_manifest_parameters["feature_id"],
                "secondary_kind": secondary_kind,
                "metadata_features": list(
                    secondary_manifest_parameters.get("input_features", [])
                ),
            }
        )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    fusion_id = (
        f"fixed-qwen-fusion-{digest}"
        if secondary_kind == "qwen_hierarchical"
        else f"fixed-qwen-metadata-fusion-{digest}"
    )
    fusion_dir = output_root.resolve() / fusion_id
    metric_summary = {
        metric: asdict(
            mean_sd_ci95(
                float(row["oof_metrics"][metric]) for row in per_repetition
            )
        )
        for metric in METRIC_NAMES
    }
    created_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": 1,
        "fusion_id": fusion_id,
        **identity,
        "created_at": created_at,
        "git": source_git,
        "statistical_unit": (
            "one complete subject-level OOF result from each split-seed repetition"
        ),
        "ci_method": (
            "two-sided 95% Student's t interval over repetition-level OOF metrics"
        ),
        "oof_subjects_per_repetition": len(reference_subjects or {}),
        "metric_summary": metric_summary,
        "per_repetition": per_repetition,
    }
    manifest = {
        "schema_version": 1,
        "fusion_id": fusion_id,
        **identity,
        "created_at": created_at,
        "git": source_git,
        "newly_trained_models": 0,
        "input_fold_models": 2 * len(per_repetition) * 5,
        "input_constraint": (
            "both endpoints consume the same "
            f"Keyword-post-{keyword_condition} Qwen3-VL post sequences"
            if secondary_kind == "qwen_hierarchical"
            else (
                f"the primary consumes Keyword-post-{keyword_condition} Qwen3-VL post sequences; "
                "the secondary consumes only fold-local standardized interaction counts "
                "and posting-time features from the aligned users and posts"
            )
        ),
    }
    fusion_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(fusion_dir / "manifest.json", manifest)
    write_jsonl_atomic(fusion_dir / "oof_predictions.jsonl", fused_rows)
    write_json_atomic(fusion_dir / "summary.json", summary)
    return fusion_dir, summary


def build_fixed_qwen_metadata_fusion(
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Fuse a Qwen sequence expert with the audited metadata-only expert."""
    return build_fixed_qwen_sequence_fusion(**kwargs, secondary_kind="metadata")
