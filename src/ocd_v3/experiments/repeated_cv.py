from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.metrics import mean_sd_ci95
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.full_config import FullExperimentConfig
from ocd_v3.experiments.train_full import TrainFullResult, train_hierarchical_full
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import read_jsonl, write_json_atomic

_METRIC_NAMES = ("f1", "roc_auc", "accuracy", "precision", "recall")


@dataclass(frozen=True)
class RepeatedCVResult:
    repeated_cv_dir: Path
    summary: dict[str, Any]
    runs: tuple[TrainFullResult, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _repetition_invariant_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    invariant = deepcopy(manifest)
    invariant.pop("created_at", None)
    invariant.pop("run_id", None)
    invariant.pop("split_id", None)
    invariant.get("seeds", {}).pop("split", None)
    return invariant


def _partition_fingerprint(subject_folds: dict[str, int]) -> str:
    """Hash an outer partition without exposing IDs or depending on fold numbering."""
    groups: dict[int, list[str]] = {}
    for subject_id, fold in subject_folds.items():
        groups.setdefault(fold, []).append(subject_id)
    canonical = sorted(sorted(subject_ids) for subject_ids in groups.values())
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def summarize_repeated_cv_runs(
    runs: Sequence[TrainFullResult],
    *,
    expected_split_seeds: Sequence[int],
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    split_seeds = tuple(int(value) for value in expected_split_seeds)
    if (
        len(runs) != len(split_seeds)
        or len(split_seeds) < 2
        or len(set(split_seeds)) != len(split_seeds)
    ):
        raise ValueError("Repeated CV requires one completed run per unique split seed")

    per_repetition: list[dict[str, Any]] = []
    reference_manifest: dict[str, Any] | None = None
    reference_subjects: dict[str, int] | None = None
    run_ids: set[str] = set()
    split_ids: set[str] = set()
    partition_fingerprints: set[str] = set()
    outer_fold_count: int | None = None

    for expected_split_seed, result in zip(split_seeds, runs, strict=True):
        run_summary = _read_json(result.run_dir / "summary.json")
        manifest = _read_json(result.run_dir / "run_manifest.json")
        if not run_summary.get("complete_outer_cv"):
            raise ValueError("Repeated-CV aggregation requires complete outer CV runs")

        run_id = str(run_summary["run_id"])
        split_id = str(run_summary["split_id"])
        if run_id != manifest.get("run_id") or run_id in run_ids:
            raise ValueError("Repeated-CV run IDs are inconsistent or duplicated")
        if split_id != manifest.get("split_id") or split_id in split_ids:
            raise ValueError("Repeated-CV split IDs are inconsistent or duplicated")
        run_ids.add(run_id)
        split_ids.add(split_id)
        if int(manifest["seeds"]["split"]) != expected_split_seed:
            raise ValueError("Run split seed differs from the requested seed order")

        invariant = _repetition_invariant_manifest(manifest)
        if reference_manifest is None:
            reference_manifest = invariant
        elif invariant != reference_manifest:
            raise ValueError("Repeated-CV runs differ outside their split identity")

        predictions = read_jsonl(result.run_dir / "oof_predictions.jsonl")
        subjects = {str(row["subject_id"]): int(row["label"]) for row in predictions}
        subject_folds = {
            str(row["subject_id"]): int(row["outer_fold"]) for row in predictions
        }
        if len(predictions) != len(subjects):
            raise ValueError("A repeated-CV run has duplicate OOF subjects")
        if reference_subjects is None:
            reference_subjects = subjects
        elif subjects != reference_subjects:
            raise ValueError("Repeated-CV runs do not cover identical subjects and labels")

        fold_ids = sorted(set(subject_folds.values()))
        if fold_ids != list(range(len(fold_ids))):
            raise ValueError("A repeated-CV run has incomplete outer-fold IDs")
        if outer_fold_count is None:
            outer_fold_count = len(fold_ids)
        elif len(fold_ids) != outer_fold_count:
            raise ValueError("Outer-fold count changed between repetitions")
        partition_sha256 = _partition_fingerprint(subject_folds)
        if partition_sha256 in partition_fingerprints:
            raise ValueError("Two split seeds produced the same outer partition")
        partition_fingerprints.add(partition_sha256)

        metrics = run_summary["oof_metrics"]
        if any(metrics.get(name) is None for name in _METRIC_NAMES):
            raise ValueError("A required repeated-CV OOF metric is undefined")
        training_seed = int(manifest["seeds"]["training_base"])
        per_repetition.append(
            {
                "split_seed": expected_split_seed,
                "split_id": split_id,
                "outer_partition_sha256": partition_sha256,
                "training_base_seed": training_seed,
                "fold_training_seeds": [
                    training_seed + fold for fold in range(len(fold_ids))
                ],
                "run_id": run_id,
                "oof_metrics": metrics,
            }
        )

    assert reference_manifest is not None
    assert reference_subjects is not None
    assert outer_fold_count is not None
    metric_summary = {
        name: asdict(
            mean_sd_ci95(
                [float(row["oof_metrics"][name]) for row in per_repetition]
            )
        )
        for name in _METRIC_NAMES
    }
    identity = {
        "dataset_id": reference_manifest["dataset_id"],
        "feature_id": reference_manifest["parameters"]["feature_id"],
        "split_seeds": list(split_seeds),
        "split_ids": [row["split_id"] for row in per_repetition],
        "run_ids": [row["run_id"] for row in per_repetition],
        "source_code": reference_manifest["source_code"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    repeated_cv_id = f"repeated-cv-{digest}"
    repeated_cv_root = output_root.resolve()
    repeated_cv_root.mkdir(parents=True, exist_ok=True)
    repeated_cv_dir = repeated_cv_root / repeated_cv_id
    repeated_cv_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "repeated_cv_id": repeated_cv_id,
        **identity,
        "experiment_id": reference_manifest.get(
            "experiment_id", "hierarchical_attention_full"
        ),
        "parameters": reference_manifest["parameters"],
        "fixed_seeds": reference_manifest["seeds"],
        "keyword_condition": reference_manifest["parameters"]["keyword_condition"],
        "classification_threshold": reference_manifest["parameters"][
            "classification_threshold"
        ],
        "training_base_seed": int(reference_manifest["seeds"]["training_base"]),
        "n_split_repetitions": len(split_seeds),
        "outer_folds_per_repetition": outer_fold_count,
        "trained_fold_models": len(split_seeds) * outer_fold_count,
        "oof_subjects_per_repetition": len(reference_subjects),
        "statistical_unit": (
            "one complete subject-level OOF result from each split-seed repetition"
        ),
        "ci_method": (
            "two-sided 95% Student's t interval over repetition-level OOF metrics"
        ),
        "interpretation_boundary": (
            "Intervals quantify partition sensitivity on this fixed dataset under one "
            "fixed fold-wise training-seed schedule. Repetitions share subjects and are "
            "not independent population samples."
        ),
        "metric_summary": metric_summary,
        "per_repetition": per_repetition,
    }
    write_json_atomic(repeated_cv_dir / "summary.json", summary)
    return repeated_cv_dir, summary


def run_repeated_full_cv(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    study_config: StudyConfig,
    full_config: FullExperimentConfig,
    run_output_root: Path,
    split_output_root: Path,
    repeated_cv_output_root: Path,
    split_seeds: Sequence[int],
    minimum_posts_per_subject: int = 1,
    include_metadata: bool = True,
    on_run_complete: Callable[[int, dict[str, Any], TrainFullResult], None]
    | None = None,
) -> RepeatedCVResult:
    seeds = tuple(int(value) for value in split_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique split seeds are required")
    if dataset.dataset_id != feature_set.dataset_id:
        raise ValueError("Dataset and feature set refer to different datasets")

    runs: list[TrainFullResult] = []
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
        result = train_hierarchical_full(
            feature_set=feature_set,
            split_assignments_path=split_path,
            study_config=study_config,
            full_config=full_config,
            output_root=run_output_root,
            experiment_id=(
                "hierarchical_attention_full"
                if include_metadata
                and getattr(full_config.model, "architecture_variant", "full") == "full"
                else "hierarchical_ablation_"
                f"{getattr(full_config.model, 'architecture_variant', 'full')}_"
                f"{'with_metadata' if include_metadata else 'without_metadata'}"
            ),
            include_metadata=include_metadata,
        )
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(split_seed, split_summary, result)

    repeated_cv_dir, summary = summarize_repeated_cv_runs(
        runs,
        expected_split_seeds=seeds,
        output_root=repeated_cv_output_root,
    )
    return RepeatedCVResult(
        repeated_cv_dir=repeated_cv_dir,
        summary=summary,
        runs=tuple(runs),
    )
