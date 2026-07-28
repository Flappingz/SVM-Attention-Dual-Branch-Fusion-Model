"""Fold-local metadata-only logistic-regression ablation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.repeated_oof import summarize_repeated_oof_runs
from ocd_v3.features.training_data import PreparedFeatureSet, fit_metadata_standardizer
from ocd_v3.io import write_json_atomic, write_jsonl_atomic
from ocd_v3.provenance import create_run_manifest, save_run_manifest

METADATA_LOGISTIC_EXPERIMENT_ID = "metadata_only_mean_logistic_regression"


@dataclass(frozen=True)
class MetadataLogisticConfig:
    c: float = 1.0
    solver: str = "liblinear"
    class_weight: str | None = None
    maximum_iterations: int = 1_000
    tolerance: float = 1e-4
    random_state: int = 42

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetadataLogisticResult:
    run_dir: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class RepeatedMetadataLogisticResult:
    repeated_cv_dir: Path
    summary: dict[str, Any]
    runs: tuple[MetadataLogisticResult, ...]


def _load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assignments"), list):
        raise ValueError("Unsupported split assignment artifact")
    return payload


def _fold_summary(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        values = [float(row[metric]) for row in fold_metrics if row[metric] is not None]
        if len(values) != len(fold_metrics):
            raise ValueError(f"Fold metric {metric} is undefined")
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


def _role_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        role: [row for row in rows if str(row["role"]) == role]
        for role in ("train", "validation", "test")
    }
    if any(not rows for rows in grouped.values()):
        raise ValueError("Every fold needs non-empty train, validation, and test roles")
    return grouped


def _subject_vector(
    feature_set: PreparedFeatureSet, subject_id: str, standardizer: Any
) -> Any:
    import numpy as np

    metadata = feature_set.load(subject_id)["metadata"]
    if metadata.shape[0] == 0:
        return np.zeros((feature_set.schema.metadata_dimension,), dtype=np.float64)
    transformed = np.asarray(
        [standardizer.transform(row) for row in metadata], dtype=np.float64
    )
    return transformed.mean(axis=0)


def train_metadata_logistic(
    *,
    feature_set: PreparedFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    output_root: Path,
    configuration: MetadataLogisticConfig | None = None,
    selected_folds: Iterable[int] | None = None,
) -> MetadataLogisticResult:
    """Fit only on fold-local standardized 5-D post metadata means."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    configuration = configuration or MetadataLogisticConfig()
    if configuration.c <= 0 or configuration.maximum_iterations < 1:
        raise ValueError("Invalid metadata logistic-regression configuration")
    split = _load_split(split_assignments_path.resolve())
    if str(split["dataset_id"]) != feature_set.dataset_id:
        raise ValueError("Feature set and split artifact refer to different datasets")
    if feature_set.schema.metadata_dimension < 1:
        raise ValueError("Metadata-only ablation requires non-empty metadata features")
    all_fold_ids = sorted({int(row["outer_fold"]) for row in split["assignments"]})
    fold_ids = all_fold_ids if selected_folds is None else sorted(set(selected_folds))
    if not fold_ids or any(fold not in all_fold_ids for fold in fold_ids):
        raise ValueError("selected_folds contains an unknown or empty fold selection")

    parameters = {
        "baseline": "metadata_only_logistic_regression",
        "configuration": configuration.public_dict(),
        "feature_id": feature_set.feature_id,
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "input_features": feature_set.manifest.get("metadata_features", []),
        "input_unit": (
            "per-subject arithmetic mean of per-post metadata after a scaler fitted "
            "only on posts from outer-fold training subjects"
        ),
        "selected_folds": fold_ids,
        "classification_threshold": study_config.evaluation.classification_threshold,
        "threshold_protocol": "fixed_from_study_config",
        "validation_protocol": "held_out_audit_only_no_model_or_threshold_selection",
        "cohort_eligibility": {
            "minimum_available_posts_per_subject": int(
                split.get("minimum_available_posts_per_subject", 1)
            ),
            "subjects": len(
                {str(row["subject_id"]) for row in split["assignments"]}
            ),
        },
    }
    repository = Path(__file__).resolve().parents[3]
    run_manifest = create_run_manifest(
        repository=repository,
        dataset_id=feature_set.dataset_id,
        split_id=str(split["split_id"]),
        experiment_id=METADATA_LOGISTIC_EXPERIMENT_ID,
        parameters=parameters,
        seeds={"split": int(split["seed"]), "estimator": configuration.random_state},
    )
    run_dir = output_root.resolve() / str(run_manifest["run_id"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return MetadataLogisticResult(
            run_dir=run_dir,
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("An incomplete run directory already exists; inspect it first")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(run_dir / "run_manifest.json", run_manifest)

    threshold = study_config.evaluation.classification_threshold
    fold_summaries: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    for outer_fold in fold_ids:
        fold_dir = run_dir / f"fold-{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            row for row in split["assignments"] if int(row["outer_fold"]) == outer_fold
        ]
        by_role = _role_rows(rows)
        training_ids = [str(row["subject_id"]) for row in by_role["train"]]
        standardizer = fit_metadata_standardizer(feature_set, training_ids)
        write_json_atomic(
            fold_dir / "metadata_standardizer.json",
            {"mean": list(standardizer.mean), "scale": list(standardizer.scale)},
        )
        matrices = {
            role: np.stack(
                [
                    _subject_vector(feature_set, str(row["subject_id"]), standardizer)
                    for row in role_rows
                ]
            )
            for role, role_rows in by_role.items()
        }
        train_labels = [int(row["label_id"]) for row in by_role["train"]]
        if len(set(train_labels)) != 2:
            raise ValueError("Each training fold must contain both classes")
        classifier = LogisticRegression(
            C=configuration.c,
            solver=configuration.solver,
            class_weight=configuration.class_weight,
            max_iter=configuration.maximum_iterations,
            tol=configuration.tolerance,
            random_state=configuration.random_state,
        )
        classifier.fit(matrices["train"], train_labels)
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

        predictions: list[dict[str, Any]] = []
        role_metrics: dict[str, dict[str, Any]] = {}
        for role, role_rows in by_role.items():
            decisions = classifier.decision_function(matrices[role]).tolist()
            scores = classifier.predict_proba(matrices[role])[:, 1].tolist()
            labels = [int(row["label_id"]) for row in role_rows]
            role_metrics[role] = binary_metrics(labels, scores, threshold).to_dict()
            role_predictions = [
                {
                    "subject_id": str(row["subject_id"]),
                    "outer_fold": outer_fold,
                    "role": role,
                    "label": int(row["label_id"]),
                    "decision_value": float(decision),
                    "score": float(score),
                    "prediction": int(score >= threshold),
                    "threshold": threshold,
                }
                for row, decision, score in zip(
                    role_rows, decisions, scores, strict=True
                )
            ]
            predictions.extend(role_predictions)
            if role == "test":
                all_oof_rows.extend(role_predictions)
        write_jsonl_atomic(fold_dir / "predictions.jsonl", predictions)
        fold_summary = {
            "outer_fold": outer_fold,
            "role_counts": {role: len(role_rows) for role, role_rows in by_role.items()},
            "metadata_dimension": feature_set.schema.metadata_dimension,
            "model_parameter_count": int(
                classifier.coef_.size + classifier.intercept_.size
            ),
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
        "feature_id": feature_set.feature_id,
        "split_id": split["split_id"],
        "keyword_condition": feature_set.manifest["keyword_condition"],
        "metadata_condition": "only",
        "complete_outer_cv": complete_outer_cv,
        "selected_folds": fold_ids,
        "threshold": threshold,
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
    return MetadataLogisticResult(run_dir=run_dir, summary=summary)


def run_repeated_metadata_logistic_cv(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    study_config: StudyConfig,
    run_output_root: Path,
    split_output_root: Path,
    repeated_cv_output_root: Path,
    split_seeds: Sequence[int],
    configuration: MetadataLogisticConfig | None = None,
    minimum_posts_per_subject: int = 1,
    on_run_complete: Callable[[int, dict[str, Any], MetadataLogisticResult], None]
    | None = None,
) -> RepeatedMetadataLogisticResult:
    seeds = tuple(int(seed) for seed in split_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique split seeds are required")
    if dataset.dataset_id != feature_set.dataset_id:
        raise ValueError("Dataset and feature set refer to different datasets")
    configuration = configuration or MetadataLogisticConfig()
    runs: list[MetadataLogisticResult] = []
    for split_seed in seeds:
        split_path, split_summary = create_split_artifact(
            dataset=dataset,
            fold_count=study_config.evaluation.outer_folds,
            validation_fraction=(
                study_config.evaluation.validation_fraction_within_outer_train
            ),
            seed=split_seed,
            output_root=split_output_root,
            minimum_posts_per_subject=minimum_posts_per_subject,
        )
        result = train_metadata_logistic(
            feature_set=feature_set,
            split_assignments_path=split_path,
            study_config=study_config,
            output_root=run_output_root,
            configuration=configuration,
        )
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(split_seed, split_summary, result)
    repeated_cv_dir, summary = summarize_repeated_oof_runs(
        runs,
        expected_split_seeds=seeds,
        output_root=repeated_cv_output_root,
        expected_experiment_id=METADATA_LOGISTIC_EXPERIMENT_ID,
        repeated_cv_id_prefix="repeated-metadata-logistic",
        interpretation_boundary=(
            "Intervals quantify partition sensitivity on this fixed cohort for a "
            "fold-local standardized metadata-only model. Repetitions reuse subjects "
            "and are not independent population samples."
        ),
    )
    return RepeatedMetadataLogisticResult(
        repeated_cv_dir=repeated_cv_dir,
        summary=summary,
        runs=tuple(runs),
    )
