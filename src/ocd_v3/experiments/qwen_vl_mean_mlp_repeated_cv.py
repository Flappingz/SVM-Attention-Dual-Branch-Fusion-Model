"""Repeated user-level CV for the Qwen3-VL mean-pooling MLP baseline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.qwen_vl_mean_mlp import (
    QWEN_VL_MEAN_MLP_EXPERIMENT_ID,
    train_qwen_vl_mean_mlp,
)
from ocd_v3.experiments.qwen_vl_mean_mlp_config import QwenVLMeanMLPConfig
from ocd_v3.experiments.repeated_oof import summarize_repeated_oof_runs
from ocd_v3.experiments.train_full import TrainFullResult
from ocd_v3.features.training_data import PreparedFeatureSet


@dataclass(frozen=True)
class RepeatedQwenVLMeanMLPResult:
    repeated_cv_dir: Path
    summary: dict[str, Any]
    runs: tuple[TrainFullResult, ...]


def run_repeated_qwen_vl_mean_mlp_cv(
    *,
    dataset: PreparedDataset,
    feature_set: PreparedFeatureSet,
    study_config: StudyConfig,
    configuration: QwenVLMeanMLPConfig,
    run_output_root: Path,
    split_output_root: Path,
    repeated_cv_output_root: Path,
    split_seeds: Sequence[int],
    minimum_posts_per_subject: int = 1,
    on_run_complete: Callable[[int, dict[str, Any], TrainFullResult], None]
    | None = None,
) -> RepeatedQwenVLMeanMLPResult:
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
        result = train_qwen_vl_mean_mlp(
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
        expected_experiment_id=QWEN_VL_MEAN_MLP_EXPERIMENT_ID,
        repeated_cv_id_prefix="repeated-qwen-vl-mean-mlp",
        interpretation_boundary=(
            "Intervals quantify partition sensitivity on this fixed dataset with frozen "
            "Qwen3-VL embeddings, masked post-mean pooling, and a fixed MLP protocol. "
            "Repetitions share subjects and are not independent population samples."
        ),
    )
    return RepeatedQwenVLMeanMLPResult(
        repeated_cv_dir=repeated_cv_dir,
        summary=summary,
        runs=tuple(runs),
    )
