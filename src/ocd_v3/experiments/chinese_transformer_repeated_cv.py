"""Repeated user-level CV for the frozen Chinese Transformer baseline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.chinese_transformer import (
    ChineseTransformerBaselineResult,
    train_chinese_transformer_baseline,
)
from ocd_v3.experiments.chinese_transformer_config import (
    ChineseTransformerBaselineConfig,
)
from ocd_v3.experiments.repeated_oof import summarize_repeated_oof_runs
from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet


@dataclass(frozen=True)
class RepeatedChineseTransformerResult:
    repeated_cv_dir: Path
    summary: dict[str, Any]
    runs: tuple[ChineseTransformerBaselineResult, ...]


def run_repeated_chinese_transformer_cv(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedChineseTransformerFeatureSet,
    study_config: StudyConfig,
    configuration: ChineseTransformerBaselineConfig,
    run_output_root: Path,
    split_output_root: Path,
    repeated_cv_output_root: Path,
    split_seeds: Sequence[int],
    minimum_posts_per_subject: int = 1,
    on_run_complete: Callable[
        [int, dict[str, Any], ChineseTransformerBaselineResult], None
    ]
    | None = None,
) -> RepeatedChineseTransformerResult:
    seeds = tuple(int(value) for value in split_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two unique split seeds are required")
    if dataset.dataset_id != feature_set.dataset_id:
        raise ValueError("Dataset and feature set refer to different datasets")
    if not feature_set.manifest.get("complete_dataset"):
        raise ValueError("Repeated CV requires a complete Transformer feature set")

    runs: list[ChineseTransformerBaselineResult] = []
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
        result = train_chinese_transformer_baseline(
            feature_set=feature_set,
            split_assignments_path=split_path,
            study_config=study_config,
            configuration=configuration,
            output_root=run_output_root,
        )
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(split_seed, split_summary, result)

    repeated_cv_dir, summary = summarize_repeated_oof_runs(
        runs,
        expected_split_seeds=seeds,
        output_root=repeated_cv_output_root,
        expected_experiment_id=(
            "frozen_chinese_roberta_mean_pool_logistic_regression"
        ),
        repeated_cv_id_prefix="repeated-chinese-transformer",
        interpretation_boundary=(
            "Intervals quantify partition sensitivity on this fixed dataset with a "
            "fixed frozen encoder and classifier configuration. Repetitions share "
            "subjects and are not independent population samples."
        ),
    )
    return RepeatedChineseTransformerResult(
        repeated_cv_dir=repeated_cv_dir,
        summary=summary,
        runs=tuple(runs),
    )
