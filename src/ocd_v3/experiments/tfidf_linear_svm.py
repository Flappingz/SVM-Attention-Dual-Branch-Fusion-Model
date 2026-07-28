"""Leakage-resistant user-level TF-IDF plus linear SVM baseline."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import KEYWORD_CONDITIONS, PreparedDataset
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof
from ocd_v3.features.text import contains_keywords, count_keyword_matches
from ocd_v3.io import write_json_atomic, write_jsonl_atomic
from ocd_v3.provenance import create_run_manifest, save_run_manifest


@dataclass(frozen=True)
class TfidfLinearSVMConfig:
    """Fixed settings for the text baseline.

    Character n-grams avoid introducing an unversioned Chinese segmentation
    dependency.  The settings are fixed before outer-CV evaluation; the
    validation role is retained for audit predictions but does not select C or
    the classification threshold.
    """

    analyzer: str = "char"
    ngram_range: tuple[int, int] = (1, 2)
    minimum_document_frequency: int = 2
    maximum_document_frequency: float = 1.0
    sublinear_tf: bool = True
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    svm_c: float = 1.0
    class_weight: str | None = None
    loss: str = "squared_hinge"
    tolerance: float = 1e-4
    maximum_iterations: int = 10_000
    random_state: int = 42

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TfidfLinearSVMResult:
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


def _sigmoid(values: Sequence[float]) -> list[float]:
    scores: list[float] = []
    for value in values:
        if not math.isfinite(value):
            raise FloatingPointError("Linear SVM produced a non-finite decision value")
        if value >= 0:
            scores.append(1.0 / (1.0 + math.exp(-value)))
        else:
            exponent = math.exp(value)
            scores.append(exponent / (1.0 + exponent))
    return scores


def _role_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        role: [row for row in rows if str(row["role"]) == role]
        for role in ("train", "validation", "test")
    }
    if any(not role_rows for role_rows in grouped.values()):
        raise ValueError("Every outer fold must have non-empty train, validation, and test roles")
    return grouped


def _subject_texts(
    *,
    dataset: PreparedDataset,
    assignments: Sequence[dict[str, Any]],
    study_config: StudyConfig,
    keyword_condition: str,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    if keyword_condition not in KEYWORD_CONDITIONS:
        raise ValueError("Unsupported keyword condition")
    expected_selected_counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    for row in assignments:
        subject_id = str(row["subject_id"])
        expected_selected_counts.setdefault(subject_id, int(row["selected_post_count"]))
        if expected_selected_counts[subject_id] != int(row["selected_post_count"]):
            raise ValueError("A subject has inconsistent selected-post counts across split folds")
        labels.setdefault(subject_id, int(row["label_id"]))
        if labels[subject_id] != int(row["label_id"]):
            raise ValueError("A subject has inconsistent labels across split folds")

    texts: dict[str, str] = {}
    audit: dict[str, dict[str, int]] = {}
    for subject_id in sorted(expected_selected_counts):
        source_posts = dataset.selected_posts(subject_id)
        posts = dataset.selected_posts(
            subject_id,
            keyword_condition=keyword_condition,
            keywords=study_config.dataset.keywords,
            replacement=study_config.dataset.keyword_mask,
        )
        if keyword_condition in {"original", "masked"} and len(posts) != expected_selected_counts[
            subject_id
        ]:
            raise ValueError("Dataset post selection no longer matches the fixed split artifact")
        if keyword_condition in {"original", "masked"} and [
            str(post["post_id"]) for post in posts
        ] != [str(post["post_id"]) for post in source_posts]:
            raise ValueError("Original and masked conditions must use identical selected posts")
        text = "\n".join(str(post["cleaned_text"]) for post in posts)
        texts[subject_id] = text
        source_texts = [str(post["cleaned_text"]) for post in source_posts]
        audit[subject_id] = {
            "source_selected_post_count": expected_selected_counts[subject_id],
            "condition_post_count": len(posts),
            "character_count": len(text),
            "empty_text": int(not text),
            "keyword_matched_post_count": sum(
                contains_keywords(value, study_config.dataset.keywords) for value in source_texts
            ),
            "keyword_match_occurrences": sum(
                count_keyword_matches(value, study_config.dataset.keywords)
                for value in source_texts
            ),
        }
    return texts, audit


def _prediction_rows(
    *,
    rows: Sequence[dict[str, Any]],
    decision_values: Sequence[float],
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
            "decision_value": float(decision_value),
            "score": float(score),
            "prediction": int(score >= threshold),
            "threshold": threshold,
        }
        for row, decision_value, score in zip(rows, decision_values, scores, strict=True)
    ]


def _save_fold_model(
    *,
    fold_dir: Path,
    vectorizer: Any,
    classifier: Any,
    configuration: TfidfLinearSVMConfig,
) -> None:
    """Save a reconstructable non-pickle TF-IDF/SVM checkpoint."""
    import numpy as np

    feature_names = [str(value) for value in vectorizer.get_feature_names_out().tolist()]
    write_json_atomic(
        fold_dir / "vectorizer.json",
        {
            "schema_version": 1,
            "configuration": configuration.public_dict(),
            "fitted_on_role": "train",
            "feature_count": len(feature_names),
            "tokens_by_index": feature_names,
        },
    )
    model_path = fold_dir / "model_parameters.npz"
    temporary = model_path.parent / f".{model_path.stem}.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.asarray([1], dtype=np.int16),
        idf=np.asarray(vectorizer.idf_, dtype=np.float64),
        coefficient=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        classes=np.asarray(classifier.classes_, dtype=np.int8),
    )
    temporary.replace(model_path)


def train_tfidf_linear_svm(
    *,
    dataset: PreparedDataset,
    split_assignments_path: Path,
    study_config: StudyConfig,
    output_root: Path,
    keyword_condition: str = "original",
    configuration: TfidfLinearSVMConfig | None = None,
    selected_folds: Iterable[int] | None = None,
) -> TfidfLinearSVMResult:
    """Run the fixed user-level five-fold TF-IDF + Linear SVM baseline."""
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    configuration = configuration or TfidfLinearSVMConfig()
    if keyword_condition not in KEYWORD_CONDITIONS:
        raise ValueError("keyword_condition must be original, masked, or removed")
    split = _load_split(split_assignments_path.resolve())
    if str(split["dataset_id"]) != dataset.dataset_id:
        raise ValueError("Dataset and split artifact refer to different datasets")
    all_fold_ids = sorted({int(row["outer_fold"]) for row in split["assignments"]})
    fold_ids = all_fold_ids if selected_folds is None else sorted(set(selected_folds))
    if not fold_ids or any(fold not in all_fold_ids for fold in fold_ids):
        raise ValueError("selected_folds contains an unknown or empty fold selection")

    texts, subject_text_audit = _subject_texts(
        dataset=dataset,
        assignments=split["assignments"],
        study_config=study_config,
        keyword_condition=keyword_condition,
    )
    parameters = {
        "baseline": "tfidf_linear_svm",
        "configuration": configuration.public_dict(),
        "input_unit": "one document per subject created by newline-joining selected posts",
        "post_selection": dataset.manifest["post_selection_for_models"],
        "keyword_condition": keyword_condition,
        "keyword_policy": {
            "match_semantics": "case_insensitive_substring",
            "keywords": list(study_config.dataset.keywords),
            "replacement": study_config.dataset.keyword_mask,
        },
        "selected_folds": fold_ids,
        "threshold_protocol": "fixed_from_study_config",
        "classification_threshold": study_config.evaluation.classification_threshold,
        "validation_protocol": (
            "retained as a held-out audit role; no hyperparameter or threshold selection"
        ),
        "score_protocol": "sigmoid_of_linear_svm_decision_value_not_calibrated_probability",
        "text_input_audit_schema_version": 2,
        "cohort_eligibility": {
            "minimum_available_posts_per_subject": int(
                split.get("minimum_available_posts_per_subject", 1)
            ),
            "subjects": len(texts),
        },
        "split_protocol": {
            key: split.get(key)
            for key in (
                "split_protocol",
                "strategy",
                "train_ratio",
                "validation_ratio",
                "test_ratio",
                "post_count_bins",
            )
            if key in split
        },
    }
    repository = Path(__file__).resolve().parents[3]
    run_manifest = create_run_manifest(
        repository=repository,
        dataset_id=dataset.dataset_id,
        split_id=str(split["split_id"]),
        experiment_id="tfidf_linear_svm",
        parameters=parameters,
        seeds={"split": int(split["seed"]), "estimator": configuration.random_state},
    )
    run_dir = output_root.resolve() / str(run_manifest["run_id"])
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return TfidfLinearSVMResult(
            run_dir=run_dir,
            summary=json.loads(summary_path.read_text(encoding="utf-8")),
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("An incomplete run directory already exists; inspect it before retrying")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(run_dir / "run_manifest.json", run_manifest)
    write_json_atomic(
        run_dir / "text_input_audit.json",
        {
            "schema_version": 2,
            "keyword_condition": keyword_condition,
            "subjects": len(subject_text_audit),
            "aggregate": {
                "source_selected_posts": sum(
                    row["source_selected_post_count"] for row in subject_text_audit.values()
                ),
                "condition_posts": sum(
                    row["condition_post_count"] for row in subject_text_audit.values()
                ),
                "characters": sum(row["character_count"] for row in subject_text_audit.values()),
                "empty_text_subjects": sum(
                    row["empty_text"] for row in subject_text_audit.values()
                ),
                "keyword_matched_posts": sum(
                    row["keyword_matched_post_count"] for row in subject_text_audit.values()
                ),
                "keyword_match_occurrences": sum(
                    row["keyword_match_occurrences"] for row in subject_text_audit.values()
                ),
            },
            "per_subject": subject_text_audit,
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
        train_texts = [texts[str(row["subject_id"])] for row in train_rows]
        train_labels = [int(row["label_id"]) for row in train_rows]
        if len(set(train_labels)) != 2:
            raise ValueError("Each training fold must contain both classes")
        vectorizer = TfidfVectorizer(
            analyzer=configuration.analyzer,
            ngram_range=configuration.ngram_range,
            min_df=configuration.minimum_document_frequency,
            max_df=configuration.maximum_document_frequency,
            sublinear_tf=configuration.sublinear_tf,
            norm=configuration.norm,
            use_idf=configuration.use_idf,
            smooth_idf=configuration.smooth_idf,
            dtype=np.float64,
        )
        train_matrix = vectorizer.fit_transform(train_texts)
        classifier = LinearSVC(
            C=configuration.svm_c,
            class_weight=configuration.class_weight,
            loss=configuration.loss,
            tol=configuration.tolerance,
            max_iter=configuration.maximum_iterations,
            dual="auto",
            random_state=configuration.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(train_matrix, train_labels)
        _save_fold_model(
            fold_dir=fold_dir,
            vectorizer=vectorizer,
            classifier=classifier,
            configuration=configuration,
        )

        prediction_rows: list[dict[str, Any]] = []
        role_metrics: dict[str, dict[str, Any]] = {}
        for role, role_rows in by_role.items():
            role_matrix = vectorizer.transform([texts[str(row["subject_id"])] for row in role_rows])
            decisions = classifier.decision_function(role_matrix).tolist()
            scores = _sigmoid([float(value) for value in decisions])
            labels = [int(row["label_id"]) for row in role_rows]
            role_metrics[role] = binary_metrics(labels, scores, threshold).to_dict()
            role_predictions = _prediction_rows(
                rows=role_rows,
                decision_values=[float(value) for value in decisions],
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
            "tfidf_feature_count": int(train_matrix.shape[1]),
            "tfidf_nonzero_train_entries": int(train_matrix.nnz),
            "model_parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
            "svm_iterations": [int(value) for value in np.atleast_1d(classifier.n_iter_)],
            "metrics": role_metrics,
        }
        write_json_atomic(fold_dir / "metrics.json", fold_summary)
        fold_summaries.append(fold_summary)

    write_jsonl_atomic(run_dir / "oof_predictions.jsonl", all_oof_rows)
    complete_outer_cv = fold_ids == all_fold_ids
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "dataset_id": dataset.dataset_id,
        "split_id": split["split_id"],
        "keyword_condition": keyword_condition,
        "complete_outer_cv": complete_outer_cv,
        "selected_folds": fold_ids,
        "threshold": threshold,
        "threshold_protocol": "fixed_before_outer_cv_evaluation",
        "validation_protocol": "held_out_audit_only_no_hyperparameter_or_threshold_selection",
        "score_protocol": "sigmoid_of_linear_svm_decision_value_not_calibrated_probability",
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
    return TfidfLinearSVMResult(run_dir=run_dir, summary=summary)
