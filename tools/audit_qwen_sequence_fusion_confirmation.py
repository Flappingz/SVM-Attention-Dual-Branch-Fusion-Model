"""Audit the Qwen sequence fusion against same-split TF-IDF."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from audit_full_linear_head_repeated_cv import _paired_comparison

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.experiments.repeated_oof import METRIC_NAMES
from ocd_v3.features.qwen_sequence import qwen_temporal_pyramid2
from ocd_v3.features.text import contains_keywords
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import read_jsonl

EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EXPECTED_REVISION = "df0de7617d1dcc3bc263f1ff3c3aa27e4172ba8f"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Mismatch for {name}: {actual} != {expected}")


def _indexed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["subject_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate OOF subject: {path}")
    return indexed


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _audit_removed_feature_sequences(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    keywords: tuple[str, ...],
) -> dict[str, int]:
    source_posts = 0
    retained_posts = 0
    removed_posts = 0
    subjects = 0
    for path in sorted(feature_set.bundle_dir.glob("*.npz")):
        subject_id = path.stem
        source = dataset.selected_posts(subject_id)
        removed = dataset.selected_posts(
            subject_id,
            keyword_condition="removed",
            keywords=keywords,
        )
        if any(
            contains_keywords(str(post["cleaned_text"]), keywords)
            for post in removed
        ):
            raise ValueError("A retained post still contains a configured keyword")
        bundle = feature_set.load(subject_id)
        if [str(value) for value in bundle["post_ids"]] != [
            str(post["post_id"]) for post in removed
        ]:
            raise ValueError("Qwen feature post IDs differ from the removed sequence")
        source_posts += len(source)
        retained_posts += len(removed)
        removed_posts += len(source) - len(removed)
        subjects += 1
    manifest_counts = feature_set.manifest["counts"]
    if retained_posts != int(manifest_counts["posts"]):
        raise ValueError("Retained post count differs from the feature manifest")
    if removed_posts != int(
        manifest_counts["keyword_matched_posts_removed_from_model_window"]
    ):
        raise ValueError("Removed post count differs from the feature manifest")
    return {
        "subjects": subjects,
        "source_window_posts": source_posts,
        "keyword_posts_removed": removed_posts,
        "retained_keyword_free_posts": retained_posts,
    }


def _audit_primary(
    *,
    artifact_root: Path,
    feature_set: PreparedFeatureSet,
    primary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parameters = primary["parameters"]
    configuration = parameters["configuration"]
    if parameters["keyword_condition"] != "removed":
        raise ValueError("Primary model is not Keyword-post-removed")
    if configuration["encoder"] != {
        "model_id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_REVISION,
        "representation": "final",
        "embedding_dimension": 2048,
        "normalize": True,
        "frozen": True,
    }:
        raise ValueError("Primary Qwen encoder config differs")
    if configuration["sequence_pooling"] != {
        "method": "temporal_pyramid2",
        "chronological_blocks": [
            "global_mean",
            "early_half_mean",
            "late_half_mean",
        ],
        "block_normalization": "l2",
        "final_normalization": "l2",
    }:
        raise ValueError("Primary sequence pooling differs")
    if configuration["classifier"]["c"] != 1.0:
        raise ValueError("Primary SVM C differs")

    subject_vectors: dict[str, np.ndarray] = {}
    for repetition in primary["per_repetition"]:
        split = _read_json(
            artifact_root
            / "splits"
            / str(repetition["split_id"])
            / "assignments.json"
        )
        for row in split["assignments"]:
            subject_id = str(row["subject_id"])
            if subject_id not in subject_vectors:
                bundle = feature_set.load(subject_id)
                vector = np.asarray(
                    qwen_temporal_pyramid2(
                        bundle["content_embeddings"][:, 0, :],
                        expected_embedding_dimension=2048,
                    ),
                    dtype=np.float64,
                )
                subject_vectors[subject_id] = vector / max(
                    float(np.linalg.norm(vector)), 1e-12
                )

    recomputed: list[dict[str, Any]] = []
    checkpoints = 0
    parameters_checked = 0
    for repetition in primary["per_repetition"]:
        split = _read_json(
            artifact_root
            / "splits"
            / str(repetition["split_id"])
            / "assignments.json"
        )
        run_dir = artifact_root / "runs" / str(repetition["run_id"])
        run_manifest = _read_json(run_dir / "run_manifest.json")
        if run_manifest["experiment_id"] != (
            "qwen3_vl_full_width_sequence_moments_linear_svm"
        ):
            raise ValueError("Primary run experiment ID differs")
        if run_manifest["parameters"] != parameters:
            raise ValueError("Primary run parameters differ across repetitions")
        stored = _indexed(run_dir / "oof_predictions.jsonl")
        labels: list[int] = []
        scores: list[float] = []
        for outer_fold in range(5):
            test_rows = [
                row
                for row in split["assignments"]
                if int(row["outer_fold"]) == outer_fold and row["role"] == "test"
            ]
            with np.load(
                run_dir / f"fold-{outer_fold}" / "model_parameters.npz",
                allow_pickle=False,
            ) as checkpoint:
                coefficient = checkpoint["coefficient"]
                intercept = checkpoint["intercept"]
                classes = checkpoint["classes"]
            if coefficient.shape != (1, 6144) or intercept.shape != (1,):
                raise ValueError("Primary checkpoint shape differs")
            if classes.tolist() != [0, 1]:
                raise ValueError("Primary checkpoint class order differs")
            if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
                raise ValueError("Primary checkpoint contains non-finite values")
            ids = [str(row["subject_id"]) for row in test_rows]
            matrix = np.stack([subject_vectors[subject_id] for subject_id in ids])
            decisions = matrix @ coefficient[0] + float(intercept[0])
            fold_scores = _sigmoid(decisions)
            for row, decision, score in zip(
                test_rows, decisions, fold_scores, strict=True
            ):
                observed = stored[str(row["subject_id"])]
                if int(observed["label"]) != int(row["label_id"]):
                    raise ValueError("Primary OOF label differs from split")
                _assert_close(
                    float(observed["decision_value"]),
                    float(decision),
                    "primary decision",
                )
                _assert_close(float(observed["score"]), float(score), "primary score")
                labels.append(int(row["label_id"]))
                scores.append(float(score))
            checkpoints += 1
            parameters_checked += int(coefficient.size + intercept.size)
        metrics = binary_metrics(labels, scores, 0.5).to_dict()
        for metric in METRIC_NAMES:
            _assert_close(
                float(metrics[metric]),
                float(repetition["oof_metrics"][metric]),
                f"primary/{repetition['split_seed']}/{metric}",
            )
        recomputed.append({**repetition, "oof_metrics": metrics})
    for metric in METRIC_NAMES:
        aggregate = asdict(
            mean_sd_ci95(float(row["oof_metrics"][metric]) for row in recomputed)
        )
        for key, value in aggregate.items():
            stored_value = primary["metric_summary"][metric][key]
            if key == "n":
                if value != stored_value:
                    raise ValueError("Primary aggregate n differs")
            else:
                _assert_close(
                    float(value), float(stored_value), f"primary/{metric}/{key}"
                )
    return recomputed, {
        "finite_npz_checkpoints": checkpoints,
        "checkpoint_parameters_checked": parameters_checked,
        "sequence_feature_dimension": 6144,
        "model_parameters_per_fold": 6145,
    }


def audit(
    *,
    artifact_root: Path,
    dataset_dir: Path,
    feature_dir: Path,
    study_config_path: Path,
    primary_summary_path: Path,
    secondary_summary_path: Path,
    tfidf_summary_path: Path,
    fusion_dir: Path,
    expected_feature_id: str | None = None,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    study = load_config(study_config_path.resolve())
    dataset = PreparedDataset(dataset_dir.resolve())
    feature_set = PreparedFeatureSet(feature_dir.resolve())
    primary = _read_json(primary_summary_path.resolve())
    secondary = _read_json(secondary_summary_path.resolve())
    tfidf = _read_json(tfidf_summary_path.resolve())
    fusion_manifest = _read_json(fusion_dir.resolve() / "manifest.json")
    fusion = _read_json(fusion_dir.resolve() / "summary.json")
    if (
        expected_feature_id is not None
        and feature_set.feature_id != expected_feature_id
    ):
        raise ValueError("Feature set does not match --expected-feature-id")
    if feature_set.manifest["encoder"]["model_id"] != EXPECTED_MODEL_ID:
        raise ValueError("Feature manifest is not Qwen3-VL-Embedding-2B")
    if feature_set.manifest["encoder"]["revision"] != EXPECTED_REVISION:
        raise ValueError("Feature revision differs")
    if feature_set.manifest["keyword_condition"] != "removed":
        raise ValueError("Feature set is not Keyword-post-removed")
    removed_audit = _audit_removed_feature_sequences(
        dataset=dataset,
        feature_set=feature_set,
        keywords=study.dataset.keywords,
    )
    primary_rows, primary_audit = _audit_primary(
        artifact_root=artifact_root,
        feature_set=feature_set,
        primary=primary,
    )
    if secondary.get("feature_id") != feature_set.feature_id:
        raise ValueError("Secondary full uses another feature set")
    if secondary.get("keyword_condition") != "removed":
        raise ValueError("Secondary full is not Keyword-post-removed")
    if primary["split_ids"] != secondary["split_ids"] or primary[
        "split_ids"
    ] != tfidf["split_ids"]:
        raise ValueError("Confirmation endpoints do not share split IDs")
    if fusion_manifest["fusion_id"] != fusion["fusion_id"]:
        raise ValueError("Fusion manifest and summary IDs differ")
    if (
        fusion_manifest["primary_weight"] != 0.9
        or fusion_manifest["secondary_weight"] != 0.1
        or fusion_manifest["threshold"] != 0.5
    ):
        raise ValueError("Fusion does not use the locked 90/10 rule")
    selection_provenance = str(
        fusion_manifest.get("selection_provenance", "")
    ).strip()
    if not selection_provenance:
        raise ValueError("Fusion manifest lacks selection provenance")
    if "tfidf" in json.dumps(fusion_manifest, ensure_ascii=False).lower():
        provenance = selection_provenance.lower()
        allowed_statement = "no tf-idf score or feature"
        if allowed_statement not in provenance:
            raise ValueError("Fusion manifest unexpectedly references TF-IDF input")

    fusion_rows = read_jsonl(fusion_dir.resolve() / "oof_predictions.jsonl")
    fusion_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in fusion_rows:
        fusion_by_seed.setdefault(int(row["split_seed"]), {})[
            str(row["subject_id"])
        ] = row
    secondary_by_seed = {
        int(row["split_seed"]): row for row in secondary["per_repetition"]
    }
    tfidf_by_seed = {
        int(row["split_seed"]): row for row in tfidf["per_repetition"]
    }
    recomputed_fusion: list[dict[str, Any]] = []
    tfidf_rows: list[dict[str, Any]] = []
    for primary_repetition in primary_rows:
        seed = int(primary_repetition["split_seed"])
        primary_oof = _indexed(
            artifact_root
            / "runs"
            / str(primary_repetition["run_id"])
            / "oof_predictions.jsonl"
        )
        secondary_oof = _indexed(
            artifact_root
            / "runs"
            / str(secondary_by_seed[seed]["run_id"])
            / "oof_predictions.jsonl"
        )
        tfidf_oof = _indexed(
            artifact_root
            / "runs"
            / str(tfidf_by_seed[seed]["run_id"])
            / "oof_predictions.jsonl"
        )
        if not set(primary_oof) == set(secondary_oof) == set(tfidf_oof):
            raise ValueError("Confirmation OOF subject sets differ")
        labels: list[int] = []
        scores: list[float] = []
        tfidf_labels: list[int] = []
        tfidf_scores: list[float] = []
        for subject_id in sorted(primary_oof):
            first = primary_oof[subject_id]
            second = secondary_oof[subject_id]
            baseline = tfidf_oof[subject_id]
            if not (
                int(first["label"]) == int(second["label"]) == int(baseline["label"])
                and int(first["outer_fold"])
                == int(second["outer_fold"])
                == int(baseline["outer_fold"])
            ):
                raise ValueError("Confirmation labels/folds differ")
            expected = 0.9 * float(first["score"]) + 0.1 * float(second["score"])
            stored = fusion_by_seed[seed][subject_id]
            _assert_close(float(stored["score"]), expected, "fusion score")
            if int(stored["prediction"]) != int(expected >= 0.5):
                raise ValueError("Fusion prediction differs")
            labels.append(int(first["label"]))
            scores.append(expected)
            tfidf_labels.append(int(baseline["label"]))
            tfidf_scores.append(float(baseline["score"]))
        metrics = binary_metrics(labels, scores, 0.5).to_dict()
        stored_repetition = next(
            row for row in fusion["per_repetition"] if int(row["split_seed"]) == seed
        )
        for metric in METRIC_NAMES:
            _assert_close(
                float(metrics[metric]),
                float(stored_repetition["oof_metrics"][metric]),
                f"fusion/{seed}/{metric}",
            )
        recomputed_fusion.append({**stored_repetition, "oof_metrics": metrics})
        tfidf_metrics = binary_metrics(tfidf_labels, tfidf_scores, 0.5).to_dict()
        for metric in METRIC_NAMES:
            _assert_close(
                float(tfidf_metrics[metric]),
                float(tfidf_by_seed[seed]["oof_metrics"][metric]),
                f"tfidf/{seed}/{metric}",
            )
        tfidf_rows.append(
            {**tfidf_by_seed[seed], "oof_metrics": tfidf_metrics}
        )
    for metric in METRIC_NAMES:
        aggregate = asdict(
            mean_sd_ci95(
                float(row["oof_metrics"][metric]) for row in recomputed_fusion
            )
        )
        for key, value in aggregate.items():
            stored_value = fusion["metric_summary"][metric][key]
            if key == "n":
                if value != stored_value:
                    raise ValueError("Fusion aggregate n differs")
            else:
                _assert_close(
                    float(value), float(stored_value), f"fusion/{metric}/{key}"
                )
    return {
        "status": "VERIFIED",
        "confirmation_status": "separate_confirmation_evaluation",
        "fusion_id": fusion["fusion_id"],
        "dataset_id": dataset.dataset_id,
        "feature_id": feature_set.feature_id,
        "split_seeds": primary["split_seeds"],
        "removed_sequence_audit": removed_audit,
        "primary_model_audit": primary_audit,
        "secondary_model_audit": {
            "repeated_cv_id": secondary["repeated_cv_id"],
            "expected_separate_checkpoint_audit": (
                "tools/audit_full_raw_mean_skip_repeated_cv.py"
            ),
        },
        "fusion_oof_rows_recomputed": len(fusion_rows),
        "metric_summary": fusion["metric_summary"],
        "paired_fusion_minus_tfidf": _paired_comparison(
            recomputed_fusion,
            tfidf_rows,
            direction="Qwen fusion minus TF-IDF",
        ),
        "checks": {
            "qwen3_vl_embedding_only_model_inputs": True,
            "chronological_post_sequence_primary_input": True,
            "hierarchical_post_sequence_secondary_input": True,
            "user_level_classification": True,
            "all_keyword_posts_excluded_before_embedding_sequence_read": True,
            "same_subject_labels_and_outer_folds_as_tfidf": True,
            "fixed_weights_0_9_and_0_1": True,
            "fixed_threshold_0_5": True,
            "no_tfidf_score_or_feature_in_model": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--secondary-summary", type=Path, required=True)
    parser.add_argument("--tfidf-summary", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-feature-id",
        help=(
            "Optional local feature artifact ID. When omitted, the audit "
            "checks cross-artifact consistency without requiring a private ID."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                artifact_root=args.artifact_root,
                dataset_dir=args.dataset_dir,
                feature_dir=args.feature_dir,
                study_config_path=args.study_config,
                primary_summary_path=args.primary_summary,
                secondary_summary_path=args.secondary_summary,
                tfidf_summary_path=args.tfidf_summary,
                fusion_dir=args.fusion_dir,
                expected_feature_id=args.expected_feature_id,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
