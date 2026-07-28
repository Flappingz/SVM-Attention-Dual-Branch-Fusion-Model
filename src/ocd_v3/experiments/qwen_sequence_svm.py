"""User-level Linear SVM over ordered full-width Qwen post-sequence moments."""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof
from ocd_v3.experiments.qwen_sequence_svm_config import QwenSequenceSVMConfig
from ocd_v3.features.qwen_sequence import (
    QWEN_SEGMENT_MEAN_STD2_BLOCKS,
    QWEN_SEQUENCE_BLOCKS,
    QWEN_TEMPORAL_PYRAMID_BLOCKS,
    qwen_mean_std_temporal_halves,
    qwen_segment_mean_std2,
    qwen_temporal_pyramid2,
)
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import write_json_atomic, write_jsonl_atomic
from ocd_v3.provenance import create_run_manifest, save_run_manifest

QWEN_SEQUENCE_SVM_EXPERIMENT_ID = "qwen3_vl_full_width_sequence_moments_linear_svm"
QWEN_SEQUENCE_PREFIX_SVM_EXPERIMENT_ID = "qwen3_vl_sequence_prefix_moments_linear_svm"
DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE = (
    "model family finalized on development partitions before evaluation on "
    "separate confirmation partitions"
)


@dataclass(frozen=True)
class QwenSequenceSVMResult:
    run_dir: Path
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _role_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        role: sorted(
            [row for row in rows if str(row["role"]) == role],
            key=lambda row: str(row["subject_id"]),
        )
        for role in ("train", "validation", "test")
    }
    if any(not values for values in grouped.values()):
        raise ValueError("Every fold requires train, validation, and test subjects")
    return grouped


def _sigmoid(values: Any) -> Any:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    if not np.isfinite(result).all():
        raise FloatingPointError("Linear SVM produced non-finite scores")
    return result


def _fold_summary(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        values = [float(row["metrics"]["test"][metric]) for row in folds]
        result[metric] = (
            asdict(mean_sd_ci95(values))
            if len(values) > 1
            else {
                "n": 1,
                "mean": values[0],
                "sample_sd": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        )
    return result


def _validate_inputs(
    feature_set: PreparedFeatureSet,
    configuration: QwenSequenceSVMConfig,
) -> dict[str, Any]:
    manifest = feature_set.manifest
    task_family = str(manifest.get("task_family", "ocd_keyword_removed"))
    if task_family == "ocd_keyword_removed":
        keyword_condition = manifest.get("keyword_condition")
        if keyword_condition not in {"original", "removed"}:
            raise ValueError(
                "OCD Qwen sequence SVM requires Original or Keyword-post-removed features"
            )
        removal = manifest.get("keyword_post_removal", {})
        if keyword_condition == "removed" and not removal.get("active"):
            raise ValueError("Feature manifest does not activate keyword-post removal")
        if keyword_condition == "original" and removal.get("active"):
            raise ValueError("Original feature manifest unexpectedly activates post removal")
    else:
        raise ValueError(f"Unsupported Qwen sequence task family: {task_family}")
    if feature_set.schema.representation_names != ("final",):
        raise ValueError("Qwen sequence SVM requires exactly the final representation")
    if feature_set.schema.embedding_dimension != configuration.encoder.embedding_dimension:
        raise ValueError("Feature embedding width differs from the locked full width")
    if feature_set.schema.maximum_posts != 64:
        raise ValueError("Qwen sequence SVM requires the fixed maximum of 64 posts")
    encoder = manifest.get("encoder", {})
    for name, expected in (
        ("model_id", configuration.encoder.model_id),
        ("revision", configuration.encoder.revision),
        ("normalize", configuration.encoder.normalize),
    ):
        if encoder.get(name) != expected:
            raise ValueError(f"Feature encoder {name} differs from the locked config")
    return manifest


def _sequence_transform(configuration: QwenSequenceSVMConfig) -> tuple[tuple[str, ...], Any]:
    if configuration.sequence_pooling.method == "mean_std_temporal_halves":
        return QWEN_SEQUENCE_BLOCKS, qwen_mean_std_temporal_halves
    if configuration.sequence_pooling.method == "temporal_pyramid2":
        return QWEN_TEMPORAL_PYRAMID_BLOCKS, qwen_temporal_pyramid2
    if configuration.sequence_pooling.method == "segment_mean_std2":
        return QWEN_SEGMENT_MEAN_STD2_BLOCKS, qwen_segment_mean_std2
    raise ValueError("Unsupported Qwen sequence transform")


def qwen_sequence_svm_experiment_id(
    configuration: QwenSequenceSVMConfig,
) -> str:
    prefix = configuration.sequence_pooling.embedding_prefix_dimension
    return (
        QWEN_SEQUENCE_SVM_EXPERIMENT_ID
        if prefix is None or prefix == configuration.encoder.embedding_dimension
        else QWEN_SEQUENCE_PREFIX_SVM_EXPERIMENT_ID
    )


def train_qwen_sequence_svm(
    *,
    feature_set: PreparedFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    configuration: QwenSequenceSVMConfig,
    output_root: Path,
    selected_folds: Iterable[int] | None = None,
    selection_provenance: str = DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE,
) -> QwenSequenceSVMResult:
    """Fit one train-only sequence classifier per user-level outer fold."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import LinearSVC

    feature_manifest = _validate_inputs(feature_set, configuration)
    task_family = str(feature_manifest.get("task_family", "ocd_keyword_removed"))
    keyword_condition = str(feature_manifest["keyword_condition"])
    sequence_blocks, transform = _sequence_transform(configuration)
    embedding_prefix_dimension = (
        configuration.sequence_pooling.embedding_prefix_dimension
        or configuration.encoder.embedding_dimension
    )
    sequence_feature_dimension = len(sequence_blocks) * embedding_prefix_dimension
    split = _read_json(split_assignments_path.resolve())
    if split.get("schema_version") != 1 or not isinstance(split.get("assignments"), list):
        raise ValueError("Unsupported split artifact")
    if split.get("dataset_id") != feature_set.dataset_id:
        raise ValueError("Feature set and split refer to different datasets")
    all_fold_ids = sorted({int(row["outer_fold"]) for row in split["assignments"]})
    fold_ids = all_fold_ids if selected_folds is None else sorted(set(selected_folds))
    if not fold_ids or any(value not in all_fold_ids for value in fold_ids):
        raise ValueError("selected_folds contains an unknown fold")

    subject_labels = {str(row["subject_id"]): int(row["label_id"]) for row in split["assignments"]}
    vectors: dict[str, Any] = {}
    post_counts: dict[str, int] = {}
    for subject_id in sorted(subject_labels):
        bundle = feature_set.load(subject_id)
        embeddings = bundle["content_embeddings"][:, 0, :]
        post_counts[subject_id] = int(embeddings.shape[0])
        vectors[subject_id] = transform(
            embeddings[:, :embedding_prefix_dimension],
            expected_embedding_dimension=embedding_prefix_dimension,
        )
    if any(value < 1 or value > 64 for value in post_counts.values()):
        raise ValueError("Every cohort subject must retain between 1 and 64 posts")

    threshold = study_config.evaluation.classification_threshold
    parameters = {
        "configuration": configuration.public_dict(),
        "feature_id": feature_set.feature_id,
        "task_family": task_family,
        "keyword_condition": keyword_condition,
        "input_unit": "one chronological Qwen embedding sequence per user",
        "sequence_blocks": list(sequence_blocks),
        "sequence_feature_dimension": sequence_feature_dimension,
        "metadata_condition": "excluded",
        "selected_folds": fold_ids,
        "classification_threshold": threshold,
        "threshold_protocol": "fixed_from_study_config",
        "validation_protocol": "audit_only_no_model_or_threshold_selection",
        "model_selection_provenance": selection_provenance,
    }
    if task_family == "ocd_keyword_removed":
        parameters["keyword_policy"] = feature_manifest["keyword_policy"]
        parameters["keyword_post_removal"] = feature_manifest["keyword_post_removal"]
    else:
        parameters["label_policy"] = feature_manifest["label_policy"]
        parameters["post_selection"] = feature_manifest["post_selection"]
    input_modalities = feature_manifest.get("embedding_input_modalities")
    if input_modalities is not None:
        parameters["embedding_input_modalities"] = list(input_modalities)
    repository = Path(__file__).resolve().parents[3]
    manifest = create_run_manifest(
        repository=repository,
        dataset_id=feature_set.dataset_id,
        split_id=str(split["split_id"]),
        experiment_id=qwen_sequence_svm_experiment_id(configuration),
        parameters=parameters,
        seeds={
            "split": int(split["seed"]),
            "estimator": configuration.classifier.random_state,
        },
    )
    run_dir = output_root.resolve() / str(manifest["run_id"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return QwenSequenceSVMResult(run_dir, _read_json(summary_path))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("Incomplete run directory exists; inspect it before retrying")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(run_dir / "run_manifest.json", manifest)
    write_json_atomic(
        run_dir / "sequence_input_audit.json",
        {
            "schema_version": 1,
            "feature_id": feature_set.feature_id,
            "encoder_model_id": configuration.encoder.model_id,
            "encoder_revision": configuration.encoder.revision,
            "embedding_dimension": configuration.encoder.embedding_dimension,
            "embedding_prefix_dimension": embedding_prefix_dimension,
            "task_family": task_family,
            "keyword_condition": keyword_condition,
            "chronological_sequence": True,
            "sequence_blocks": list(sequence_blocks),
            "subject_count": len(vectors),
            "post_count": {
                "minimum": min(post_counts.values()),
                "maximum": max(post_counts.values()),
                "total": sum(post_counts.values()),
            },
            "per_subject_post_count": post_counts,
            **(
                {
                    "keyword_post_removal": feature_manifest["keyword_post_removal"],
                    "keyword_matched_posts_removed_from_model_window": (
                        feature_manifest["counts"][
                            "keyword_matched_posts_removed_from_model_window"
                        ]
                    ),
                }
                if task_family == "ocd_keyword_removed"
                else {
                    "label_policy": feature_manifest["label_policy"],
                    "post_selection": feature_manifest["post_selection"],
                }
            ),
        },
    )

    fold_summaries: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for outer_fold in fold_ids:
        fold_dir = run_dir / f"fold-{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = [row for row in split["assignments"] if int(row["outer_fold"]) == outer_fold]
        by_role = _role_rows(fold_rows)
        train_ids = [str(row["subject_id"]) for row in by_role["train"]]
        train_x = np.stack([vectors[value] for value in train_ids])
        train_y = np.asarray([subject_labels[value] for value in train_ids], dtype=np.int64)
        normalizer = Normalizer(norm="l2")
        normalized_train = normalizer.fit_transform(train_x)
        classifier = LinearSVC(
            C=configuration.classifier.c,
            class_weight=configuration.classifier.class_weight,
            loss=configuration.classifier.loss,
            tol=configuration.classifier.tolerance,
            max_iter=configuration.classifier.maximum_iterations,
            dual="auto",
            random_state=configuration.classifier.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(normalized_train, train_y)
        model_path = fold_dir / "model_parameters.npz"
        temporary = model_path.parent / f".{model_path.stem}.tmp.npz"
        np.savez_compressed(
            temporary,
            schema_version=np.asarray([1], dtype=np.int16),
            coefficient=np.asarray(classifier.coef_, dtype=np.float64),
            intercept=np.asarray(classifier.intercept_, dtype=np.float64),
            classes=np.asarray(classifier.classes_, dtype=np.int8),
        )
        temporary.replace(model_path)
        write_json_atomic(
            fold_dir / "sequence_transform.json",
            {
                "schema_version": 1,
                "representation": "final",
                "embedding_dimension": configuration.encoder.embedding_dimension,
                "sequence_blocks": list(sequence_blocks),
                "block_normalization": "l2",
                "final_normalization": "l2",
                "fitted_parameters": 0,
            },
        )

        role_metrics: dict[str, dict[str, Any]] = {}
        prediction_rows: list[dict[str, Any]] = []
        for role, rows in by_role.items():
            ids = [str(row["subject_id"]) for row in rows]
            role_x = normalizer.transform(np.stack([vectors[value] for value in ids]))
            decisions = np.asarray(classifier.decision_function(role_x), dtype=np.float64)
            scores = _sigmoid(decisions)
            labels = [subject_labels[value] for value in ids]
            role_metrics[role] = binary_metrics(labels, scores.tolist(), threshold).to_dict()
            for row, decision, score in zip(rows, decisions, scores, strict=True):
                prediction = {
                    "subject_id": str(row["subject_id"]),
                    "outer_fold": outer_fold,
                    "role": role,
                    "label": int(row["label_id"]),
                    "post_count": post_counts[str(row["subject_id"])],
                    "decision_value": float(decision),
                    "score": float(score),
                    "prediction": int(score >= threshold),
                    "threshold": threshold,
                }
                prediction_rows.append(prediction)
                if role == "test":
                    oof_rows.append(prediction)
        write_jsonl_atomic(fold_dir / "predictions.jsonl", prediction_rows)
        fold_summaries.append(
            {
                "outer_fold": outer_fold,
                "role_counts": {role: len(rows) for role, rows in by_role.items()},
                "role_label_counts": {
                    role: dict(sorted(Counter(int(row["label_id"]) for row in rows).items()))
                    for role, rows in by_role.items()
                },
                "sequence_feature_dimension": int(train_x.shape[1]),
                "model_parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
                "svm_iterations": [int(value) for value in np.atleast_1d(classifier.n_iter_)],
                "metrics": role_metrics,
            }
        )

    write_jsonl_atomic(run_dir / "oof_predictions.jsonl", oof_rows)
    complete = fold_ids == all_fold_ids
    is_holdout = split.get("split_protocol") == "single_group_stratified_holdout"
    oof_metrics = None
    if complete:
        expected_subject_ids = (
            {str(row["subject_id"]) for row in split["assignments"] if str(row["role"]) == "test"}
            if is_holdout
            else set(subject_labels)
        )
        oof_metrics = evaluate_oof(
            [
                OOFPrediction(
                    subject_id=str(row["subject_id"]),
                    outer_fold=int(row["outer_fold"]),
                    label=int(row["label"]),
                    score=float(row["score"]),
                )
                for row in oof_rows
            ],
            expected_subject_ids=expected_subject_ids,
            threshold=threshold,
        ).to_dict()
    summary = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "dataset_id": feature_set.dataset_id,
        "feature_id": feature_set.feature_id,
        "split_id": split["split_id"],
        "experiment_id": qwen_sequence_svm_experiment_id(configuration),
        "task_family": task_family,
        "keyword_condition": keyword_condition,
        "complete_outer_cv": complete and not is_holdout,
        "holdout_evaluation": complete and is_holdout,
        "classification_threshold": threshold,
        "sequence_feature_dimension": sequence_feature_dimension,
        "model_parameter_count": sequence_feature_dimension + 1,
        "folds": fold_summaries,
        "fold_metric_summary": _fold_summary(fold_summaries),
    }
    summary["holdout_metrics" if is_holdout else "oof_metrics"] = oof_metrics
    if input_modalities is not None:
        summary["embedding_input_modalities"] = list(input_modalities)
    if oof_metrics is not None and any(
        not math.isfinite(float(oof_metrics[name]))
        for name in ("f1", "roc_auc", "accuracy", "precision", "recall")
    ):
        raise FloatingPointError("Non-finite OOF metric")
    write_json_atomic(summary_path, summary)
    return QwenSequenceSVMResult(run_dir, summary)
