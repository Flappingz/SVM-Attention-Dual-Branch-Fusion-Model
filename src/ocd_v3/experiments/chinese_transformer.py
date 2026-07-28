"""Frozen Chinese RoBERTa mean-pooling baseline with a train-only classifier."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof
from ocd_v3.experiments.chinese_transformer_config import (
    ChineseTransformerBaselineConfig,
)
from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet
from ocd_v3.io import write_json_atomic, write_jsonl_atomic
from ocd_v3.provenance import create_run_manifest, save_run_manifest


@dataclass(frozen=True)
class ChineseTransformerBaselineResult:
    run_dir: Path
    summary: dict[str, Any]


def _load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assignments"), list):
        raise ValueError("Unsupported split assignment artifact")
    return payload


def _fold_summary(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        values = [float(row[metric]) for row in fold_metrics if row[metric] is not None]
        if len(values) != len(fold_metrics):
            raise ValueError(f"Fold metric {metric} is undefined in at least one test fold")
        if len(values) == 1:
            summary[metric] = {
                "n": 1,
                "mean": values[0],
                "sample_sd": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        else:
            summary[metric] = asdict(mean_sd_ci95(values))
    return summary


def _role_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        role: [row for row in rows if str(row["role"]) == role]
        for role in ("train", "validation", "test")
    }
    if any(not role_rows for role_rows in grouped.values()):
        raise ValueError("Every outer fold must have non-empty train, validation, and test roles")
    return grouped


def _prediction_rows(
    *,
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    outer_fold: int,
    role: str,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "subject_id": str(row["subject_id"]),
            "outer_fold": outer_fold,
            "role": role,
            "label": int(row["label_id"]),
            "score": float(score),
            "prediction": int(score >= threshold),
            "threshold": threshold,
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def _save_fold_model(
    *,
    fold_dir: Path,
    scaler: Any,
    classifier: Any,
    configuration: ChineseTransformerBaselineConfig,
) -> None:
    import numpy as np

    scaler_path = fold_dir / "scaler_parameters.npz"
    scaler_temporary = scaler_path.parent / f".{scaler_path.stem}.tmp.npz"
    np.savez_compressed(
        scaler_temporary,
        schema_version=np.asarray([1], dtype=np.int16),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        variance=np.asarray(scaler.var_, dtype=np.float64),
        samples_seen=np.asarray([scaler.n_samples_seen_], dtype=np.int64),
    )
    scaler_temporary.replace(scaler_path)

    model_path = fold_dir / "model_parameters.npz"
    model_temporary = model_path.parent / f".{model_path.stem}.tmp.npz"
    np.savez_compressed(
        model_temporary,
        schema_version=np.asarray([1], dtype=np.int16),
        coefficient=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        classes=np.asarray(classifier.classes_, dtype=np.int8),
        iterations=np.asarray(classifier.n_iter_, dtype=np.int32),
    )
    model_temporary.replace(model_path)
    write_json_atomic(
        fold_dir / "configuration.json",
        {
            "schema_version": 1,
            "classifier_configuration": configuration.classifier.public_dict(),
            "fitted_on_role": "train",
        },
    )


def _load_subject_vectors(
    feature_set: PreparedChineseTransformerFeatureSet,
    assignments: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    import numpy as np

    expected_counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    for row in assignments:
        subject_id = str(row["subject_id"])
        expected_counts.setdefault(subject_id, int(row["selected_post_count"]))
        if expected_counts[subject_id] != int(row["selected_post_count"]):
            raise ValueError("A subject has inconsistent selected-post counts across folds")
        labels.setdefault(subject_id, int(row["label_id"]))
        if labels[subject_id] != int(row["label_id"]):
            raise ValueError("A subject has inconsistent labels across folds")

    vectors: dict[str, Any] = {}
    audit: dict[str, dict[str, int]] = {}
    keyword_condition = str(feature_set.manifest.get("keyword_condition"))
    for subject_id in sorted(expected_counts):
        payload = feature_set.load(subject_id)
        embeddings = np.asarray(payload["post_embeddings"], dtype=np.float32)
        source_count = expected_counts[subject_id]
        if keyword_condition == "removed":
            if len(embeddings) > source_count:
                raise ValueError(
                    "Removed-condition features exceed the fixed source post window"
                )
        elif len(embeddings) != source_count:
            raise ValueError("Feature post count no longer matches the fixed split artifact")
        if not len(embeddings):
            raise ValueError("Chinese Transformer mean pooling requires at least one post")
        vector = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
        if vector.shape != (feature_set.hidden_size,) or not np.isfinite(vector).all():
            raise FloatingPointError("Invalid subject-level Transformer mean embedding")
        vectors[subject_id] = vector
        audit[subject_id] = {
            "selected_post_count": len(embeddings),
            "source_selected_post_count": source_count,
            "keyword_removed_post_count": source_count - len(embeddings),
            "embedding_dimensions": len(vector),
        }
    return vectors, audit


def train_chinese_transformer_baseline(
    *,
    feature_set: PreparedChineseTransformerFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    configuration: ChineseTransformerBaselineConfig,
    output_root: Path,
    selected_folds: Iterable[int] | None = None,
) -> ChineseTransformerBaselineResult:
    """Train five train-only downstream classifiers over frozen user embeddings."""
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    split = _load_split(split_assignments_path.resolve())
    if str(split["dataset_id"]) != feature_set.dataset_id:
        raise ValueError("Feature set and split artifact refer to different datasets")
    all_fold_ids = sorted({int(row["outer_fold"]) for row in split["assignments"]})
    fold_ids = all_fold_ids if selected_folds is None else sorted(set(selected_folds))
    if not fold_ids or any(fold not in all_fold_ids for fold in fold_ids):
        raise ValueError("selected_folds contains an unknown or empty fold selection")

    vectors, input_audit = _load_subject_vectors(feature_set, split["assignments"])
    parameters = {
        "baseline": "frozen_chinese_roberta_mean_pool_logistic_regression",
        "feature_id": feature_set.feature_id,
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "encoder_configuration": configuration.encoder.public_dict(),
        "encoder_manifest": feature_set.manifest["encoder"],
        "classifier_configuration": configuration.classifier.public_dict(),
        "token_pooling": configuration.encoder.token_pooling,
        "post_pooling": configuration.encoder.post_pooling,
        "selected_folds": fold_ids,
        "classification_threshold": study_config.evaluation.classification_threshold,
        "threshold_protocol": "fixed_from_study_config",
        "validation_protocol": (
            "held-out audit only; no hyperparameter, scaler, or threshold selection"
        ),
        "split_protocol": {
            key: split.get(key)
            for key in (
                "algorithm_version",
                "minimum_available_posts_per_subject",
                "outer_folds",
                "validation_fraction",
            )
            if key in split
        },
    }
    repository = Path(__file__).resolve().parents[3]
    run_manifest = create_run_manifest(
        repository=repository,
        dataset_id=feature_set.dataset_id,
        split_id=str(split["split_id"]),
        experiment_id="frozen_chinese_roberta_mean_pool_logistic_regression",
        parameters=parameters,
        seeds={
            "split": int(split["seed"]),
            "classifier": configuration.classifier.random_state,
        },
    )
    run_dir = output_root.resolve() / str(run_manifest["run_id"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return ChineseTransformerBaselineResult(
            run_dir=run_dir,
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("An incomplete run directory already exists; inspect before retrying")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(run_dir / "run_manifest.json", run_manifest)
    write_json_atomic(
        run_dir / "text_feature_audit.json",
        {
            "schema_version": 1,
            "feature_id": feature_set.feature_id,
            "subjects": len(input_audit),
            "selected_posts": sum(row["selected_post_count"] for row in input_audit.values()),
            "embedding_dimensions": feature_set.hidden_size,
            "per_subject": input_audit,
        },
    )

    threshold = study_config.evaluation.classification_threshold
    fold_summaries: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    for outer_fold in fold_ids:
        fold_dir = run_dir / f"fold-{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = [
            row for row in split["assignments"] if int(row["outer_fold"]) == outer_fold
        ]
        by_role = _role_rows(fold_rows)
        train_rows = by_role["train"]
        train_matrix = np.stack([vectors[str(row["subject_id"])] for row in train_rows])
        train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int8)
        if set(train_labels.tolist()) != {0, 1}:
            raise ValueError("Each training fold must contain both classes")
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_scaled = scaler.fit_transform(train_matrix)
        classifier = LogisticRegression(
            C=configuration.classifier.c,
            solver=configuration.classifier.solver,
            l1_ratio=0.0,
            max_iter=configuration.classifier.maximum_iterations,
            class_weight=configuration.classifier.class_weight,
            random_state=configuration.classifier.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(train_scaled, train_labels)
        _save_fold_model(
            fold_dir=fold_dir,
            scaler=scaler,
            classifier=classifier,
            configuration=configuration,
        )
        positive_index = classifier.classes_.tolist().index(1)
        prediction_rows: list[dict[str, Any]] = []
        role_metrics: dict[str, dict[str, Any]] = {}
        for role, role_rows in by_role.items():
            matrix = np.stack([vectors[str(row["subject_id"])] for row in role_rows])
            scaled = scaler.transform(matrix)
            scores = classifier.predict_proba(scaled)[:, positive_index].astype(float).tolist()
            labels = [int(row["label_id"]) for row in role_rows]
            role_metrics[role] = binary_metrics(labels, scores, threshold).to_dict()
            role_predictions = _prediction_rows(
                rows=role_rows,
                scores=scores,
                outer_fold=outer_fold,
                role=role,
                threshold=threshold,
            )
            prediction_rows.extend(role_predictions)
            if role == "test":
                all_oof_rows.extend(role_predictions)
        write_jsonl_atomic(fold_dir / "predictions.jsonl", prediction_rows)
        fold_summary = {
            "outer_fold": outer_fold,
            "role_counts": {role: len(rows) for role, rows in by_role.items()},
            "role_label_counts": {
                role: dict(sorted(Counter(int(row["label_id"]) for row in rows).items()))
                for role, rows in by_role.items()
            },
            "encoder_parameter_count": int(feature_set.manifest["encoder"]["parameter_count"]),
            "trainable_classifier_parameter_count": int(
                classifier.coef_.size + classifier.intercept_.size
            ),
            "classifier_iterations": [int(value) for value in classifier.n_iter_],
            "scaler_fitted_subjects": len(train_rows),
            "metrics": role_metrics,
        }
        write_json_atomic(fold_dir / "metrics.json", fold_summary)
        fold_summaries.append(fold_summary)

    write_jsonl_atomic(run_dir / "oof_predictions.jsonl", all_oof_rows)
    complete_outer_cv = fold_ids == all_fold_ids
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "dataset_id": feature_set.dataset_id,
        "split_id": split["split_id"],
        "feature_id": feature_set.feature_id,
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "complete_outer_cv": complete_outer_cv,
        "selected_folds": fold_ids,
        "threshold": threshold,
        "threshold_protocol": "fixed_before_outer_cv_evaluation",
        "validation_protocol": (
            "held_out_audit_only_no_hyperparameter_scaler_or_threshold_selection"
        ),
        "score_protocol": "logistic_regression_positive_class_probability",
        "folds": fold_summaries,
    }
    if complete_outer_cv:
        expected_subject_ids = {
            str(row["subject_id"])
            for row in split["assignments"]
            if str(row["role"]) == "test"
        }
        oof_predictions = [
            OOFPrediction(
                subject_id=str(row["subject_id"]),
                outer_fold=int(row["outer_fold"]),
                label=int(row["label"]),
                score=float(row["score"]),
            )
            for row in all_oof_rows
        ]
        summary["fold_test_metrics"] = _fold_summary(
            [row["metrics"]["test"] for row in fold_summaries]
        )
        summary["oof_metrics"] = evaluate_oof(
            oof_predictions, expected_subject_ids, threshold
        ).to_dict()
    write_json_atomic(summary_path, summary)
    return ChineseTransformerBaselineResult(run_dir=run_dir, summary=summary)
