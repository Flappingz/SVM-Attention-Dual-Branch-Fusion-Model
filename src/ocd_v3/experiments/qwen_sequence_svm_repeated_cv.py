"""Repeated outer CV for the locked Qwen sequence-moments Linear SVM."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.qwen_sequence_svm import (
    DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE,
    QwenSequenceSVMResult,
    qwen_sequence_svm_experiment_id,
    train_qwen_sequence_svm,
)
from ocd_v3.experiments.qwen_sequence_svm_config import QwenSequenceSVMConfig
from ocd_v3.experiments.repeated_oof import summarize_repeated_oof_runs
from ocd_v3.features.text import contains_keywords
from ocd_v3.features.training_data import PreparedFeatureSet


@dataclass(frozen=True)
class RepeatedQwenSequenceSVMResult:
    repeated_cv_dir: Path
    summary: dict[str, Any]
    runs: tuple[QwenSequenceSVMResult, ...]


def _audit_feature_sequences(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    study_config: StudyConfig,
    subject_ids: Sequence[str],
) -> None:
    keyword_condition = str(feature_set.manifest.get("keyword_condition"))
    if keyword_condition not in {"original", "removed"}:
        raise ValueError(
            "Qwen sequence repeated CV supports only original or removed features"
        )
    for subject_id in subject_ids:
        posts = dataset.selected_posts(
            subject_id,
            keyword_condition=keyword_condition,
            keywords=study_config.dataset.keywords,
            replacement=study_config.dataset.keyword_mask,
        )
        if keyword_condition == "removed" and any(
            contains_keywords(str(post["cleaned_text"]), study_config.dataset.keywords)
            for post in posts
        ):
            raise ValueError("A supposedly removed sequence still contains a keyword post")
        bundle = feature_set.load(subject_id)
        if [str(value) for value in bundle["post_ids"]] != [
            str(post["post_id"]) for post in posts
        ]:
            raise ValueError(
                f"Qwen feature sequence differs from {keyword_condition} dataset posts"
            )


def run_repeated_qwen_sequence_svm_cv(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    study_config: StudyConfig,
    configuration: QwenSequenceSVMConfig,
    run_output_root: Path,
    split_output_root: Path,
    repeated_cv_output_root: Path,
    split_seeds: Sequence[int],
    minimum_posts_per_subject: int = 1,
    selection_provenance: str = DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE,
    interpretation_boundary: str | None = None,
    on_run_complete: Callable[[int, dict[str, Any], QwenSequenceSVMResult], None]
    | None = None,
) -> RepeatedQwenSequenceSVMResult:
    seeds = tuple(int(value) for value in split_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique split seeds are required")
    if dataset.dataset_id != feature_set.dataset_id:
        raise ValueError("Dataset and feature set refer to different datasets")
    eligible_subjects = sorted(
        subject.subject_id
        for subject in dataset.subjects()
        if subject.post_count >= minimum_posts_per_subject
    )
    _audit_feature_sequences(
        dataset=dataset,
        feature_set=feature_set,
        study_config=study_config,
        subject_ids=eligible_subjects,
    )

    runs: list[QwenSequenceSVMResult] = []
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
        result = train_qwen_sequence_svm(
            feature_set=feature_set,
            split_assignments_path=split_path,
            study_config=study_config,
            configuration=configuration,
            output_root=run_output_root,
            selection_provenance=selection_provenance,
        )
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(split_seed, split_summary, result)

    repeated_cv_dir, summary = summarize_repeated_oof_runs(
        runs,
        expected_split_seeds=seeds,
        output_root=repeated_cv_output_root,
        expected_experiment_id=qwen_sequence_svm_experiment_id(configuration),
        repeated_cv_id_prefix="repeated-qwen-sequence-moments-svm",
        interpretation_boundary=(
            interpretation_boundary
            if interpretation_boundary is not None
            else (
                f"The full-width Qwen sequence architecture and C="
                f"{configuration.classifier.c:g} were locked after an "
                "explicit split-seed 42--49 exploration.  These repetitions quantify "
                "partition sensitivity on the same fixed cohort and are not external "
                "population validation."
            )
        ),
    )
    return RepeatedQwenSequenceSVMResult(
        repeated_cv_dir=repeated_cv_dir,
        summary=summary,
        runs=tuple(runs),
    )
