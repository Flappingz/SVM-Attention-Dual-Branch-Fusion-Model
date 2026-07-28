"""Run and audit the four-model Keyword-post-removed controlled benchmark."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.evaluation.metrics import mean_sd_ci95
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.experiments.chinese_transformer import train_chinese_transformer_baseline
from ocd_v3.experiments.chinese_transformer_config import (
    ChineseTransformerBaselineConfig,
)
from ocd_v3.experiments.controlled_benchmark_config import (
    CONTROLLED_METRICS,
    ControlledBenchmarkConfig,
)
from ocd_v3.experiments.full_config import FullExperimentConfig
from ocd_v3.experiments.qwen_vl_mean_mlp import (
    QWEN_VL_MEAN_MLP_EXPERIMENT_ID,
    train_qwen_vl_mean_mlp,
)
from ocd_v3.experiments.qwen_vl_mean_mlp_config import QwenVLMeanMLPConfig
from ocd_v3.experiments.repeated_cv import summarize_repeated_cv_runs
from ocd_v3.experiments.repeated_oof import summarize_repeated_oof_runs
from ocd_v3.experiments.tfidf_linear_svm import (
    TfidfLinearSVMConfig,
    train_tfidf_linear_svm,
)
from ocd_v3.experiments.tfidf_repeated_cv import (
    summarize_repeated_tfidf_linear_svm_runs,
)
from ocd_v3.experiments.train_full import train_hierarchical_full
from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import read_jsonl, write_json_atomic
from ocd_v3.provenance import git_state


@dataclass(frozen=True)
class ControlledSplitRecord:
    seed: int
    assignments_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class ControlledBenchmarkResult:
    benchmark_dir: Path
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact_relative(path: Path, artifact_root: Path) -> str:
    return str(path.resolve().relative_to(artifact_root.resolve())).replace("\\", "/")


def _validate_protocol(
    *,
    benchmark_config: ControlledBenchmarkConfig,
    dataset: PreparedDataset,
    study_config: StudyConfig,
    qwen_feature_set: PreparedFeatureSet,
    chinese_feature_set: PreparedChineseTransformerFeatureSet,
    full_config: FullExperimentConfig,
    qwen_config: QwenVLMeanMLPConfig,
    chinese_config: ChineseTransformerBaselineConfig,
    tfidf_config: TfidfLinearSVMConfig,
) -> None:
    if (
        benchmark_config.expected_dataset_id is not None
        and dataset.dataset_id != benchmark_config.expected_dataset_id
    ):
        raise ValueError("Dataset ID differs from the controlled-benchmark protocol")
    if {qwen_feature_set.dataset_id, chinese_feature_set.dataset_id} != {
        dataset.dataset_id
    }:
        raise ValueError("Both feature sets must refer to the controlled dataset")
    if study_config.evaluation.outer_folds != benchmark_config.splits.outer_folds:
        raise ValueError("Study and benchmark outer-fold counts differ")
    if not math.isclose(
        study_config.evaluation.validation_fraction_within_outer_train,
        benchmark_config.splits.validation_fraction_within_outer_train,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Study and benchmark validation fractions differ")
    if study_config.evaluation.classification_threshold != (
        benchmark_config.classification_threshold
    ):
        raise ValueError("Study and benchmark classification thresholds differ")
    if qwen_feature_set.manifest.get("keyword_condition") != "removed":
        raise ValueError("The Qwen feature set is not Keyword-post-removed")
    if chinese_feature_set.manifest.get("keyword_condition") != "removed":
        raise ValueError("The Chinese RoBERTa feature set is not Keyword-post-removed")
    if not qwen_feature_set.manifest.get("complete_dataset"):
        raise ValueError("The Qwen feature set is incomplete")
    if not chinese_feature_set.manifest.get("complete_dataset"):
        raise ValueError("The Chinese RoBERTa feature set is incomplete")

    qwen_encoder = qwen_feature_set.manifest.get("encoder", {})
    expected_qwen = {
        "model_id": full_config.encoder.model_id,
        "revision": full_config.encoder.revision,
        "representations": list(full_config.encoder.representations),
        "normalize": full_config.encoder.normalize,
    }
    if any(qwen_encoder.get(key) != value for key, value in expected_qwen.items()):
        raise ValueError("The full-model Qwen feature encoder differs from its config")
    if (
        qwen_config.encoder.model_id != full_config.encoder.model_id
        or qwen_config.encoder.revision != full_config.encoder.revision
        or qwen_config.encoder.representations != full_config.encoder.representations
        or qwen_config.encoder.normalize != full_config.encoder.normalize
    ):
        raise ValueError("Qwen+MLP and full must use the identical frozen Qwen encoder")
    chinese_encoder = chinese_feature_set.manifest.get("encoder", {})
    if (
        chinese_encoder.get("model_id") != chinese_config.encoder.model_id
        or chinese_encoder.get("revision") != chinese_config.encoder.revision
    ):
        raise ValueError("The Chinese RoBERTa feature encoder differs from its config")

    seed = benchmark_config.training_base_seed
    if full_config.training.seed != seed or qwen_config.training.seed != seed:
        raise ValueError("Both neural models must use the frozen training base seed")
    if chinese_config.classifier.random_state != seed:
        raise ValueError("Chinese RoBERTa classifier random_state differs from the base seed")
    if tfidf_config.random_state != seed:
        raise ValueError("TF-IDF estimator random_state differs from the base seed")


def audit_controlled_inputs(
    *,
    dataset: PreparedDataset,
    study_config: StudyConfig,
    benchmark_config: ControlledBenchmarkConfig,
    qwen_feature_set: PreparedFeatureSet,
    chinese_feature_set: PreparedChineseTransformerFeatureSet,
) -> dict[str, Any]:
    """Prove cohort and post identity before any model is trained."""
    all_subjects = list(dataset.subjects())
    minimum = benchmark_config.cohort.minimum_posts_per_subject
    included = [subject for subject in all_subjects if subject.post_count >= minimum]
    excluded = [subject for subject in all_subjects if subject.post_count < minimum]
    if not included:
        raise ValueError("The controlled cohort is empty")

    total_source_posts = 0
    total_condition_posts = 0
    removed_posts = 0
    empty_subjects: list[str] = []
    for subject in included:
        source_posts = dataset.selected_posts(subject.subject_id)
        condition_posts = dataset.selected_posts(
            subject.subject_id,
            keyword_condition="removed",
            keywords=study_config.dataset.keywords,
            replacement=study_config.dataset.keyword_mask,
        )
        expected_ids = [str(row["post_id"]) for row in condition_posts]
        qwen_ids = [str(value) for value in qwen_feature_set.load(subject.subject_id)["post_ids"]]
        chinese_ids = [
            str(value) for value in chinese_feature_set.load(subject.subject_id)["post_ids"]
        ]
        if qwen_ids != expected_ids or chinese_ids != expected_ids:
            raise ValueError(
                f"Feature post IDs/order differ from controlled input for {subject.subject_id}"
            )
        total_source_posts += len(source_posts)
        total_condition_posts += len(condition_posts)
        removed_posts += len(source_posts) - len(condition_posts)
        if not condition_posts:
            empty_subjects.append(subject.subject_id)
    if empty_subjects:
        raise ValueError("Keyword removal left controlled-cohort subjects without model posts")

    return {
        "rule": {
            "field": "available_post_count",
            "operator": ">",
            "threshold": benchmark_config.cohort.threshold,
            "split_api_equivalent_minimum": minimum,
        },
        "subjects_before_filter": len(all_subjects),
        "subjects_included": len(included),
        "subjects_excluded": len(excluded),
        "included_labels": dict(
            sorted(Counter(subject.label_name for subject in included).items())
        ),
        "excluded_labels": dict(
            sorted(Counter(subject.label_name for subject in excluded).items())
        ),
        "source_model_window_posts": total_source_posts,
        "condition_posts": total_condition_posts,
        "keyword_matched_posts_removed": removed_posts,
        "subjects_without_condition_posts": 0,
        "feature_alignment": {
            "qwen_feature_id": qwen_feature_set.feature_id,
            "chinese_transformer_feature_id": chinese_feature_set.feature_id,
            "subject_post_ids_and_order_exact": True,
        },
    }


def create_controlled_splits(
    *,
    dataset: PreparedDataset,
    study_config: StudyConfig,
    benchmark_config: ControlledBenchmarkConfig,
) -> tuple[ControlledSplitRecord, ...]:
    records: list[ControlledSplitRecord] = []
    expected_cohort: dict[str, Any] | None = None
    for seed in benchmark_config.splits.seeds:
        path, summary = create_split_artifact(
            dataset=dataset,
            fold_count=benchmark_config.splits.outer_folds,
            validation_fraction=(
                benchmark_config.splits.validation_fraction_within_outer_train
            ),
            seed=seed,
            output_root=study_config.split_dir,
            minimum_posts_per_subject=(
                benchmark_config.cohort.minimum_posts_per_subject
            ),
        )
        cohort = summary["cohort"]
        if expected_cohort is None:
            expected_cohort = cohort
        elif cohort != expected_cohort:
            raise ValueError("Cohort membership changed between split seeds")
        records.append(ControlledSplitRecord(seed=seed, assignments_path=path, summary=summary))
    return tuple(records)


def _expected_oof_identity(split_path: Path) -> dict[str, tuple[int, int]]:
    split = _read_json(split_path)
    result: dict[str, tuple[int, int]] = {}
    for row in split["assignments"]:
        if row["role"] != "test":
            continue
        subject_id = str(row["subject_id"])
        if subject_id in result:
            raise ValueError("A split assigns one subject to test more than once")
        result[subject_id] = (int(row["label_id"]), int(row["outer_fold"]))
    return result


def audit_model_oof_alignment(
    *,
    split_records: Sequence[ControlledSplitRecord],
    runs_by_model: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    if set(runs_by_model) != {
        "tfidf_linear_svm",
        "chinese_roberta_mean_logistic",
        "qwen_vl_mean_mlp",
        "hierarchical_attention_full",
    }:
        raise ValueError("OOF audit requires exactly the four controlled models")
    expected_split_ids = [str(record.summary["split_id"]) for record in split_records]
    prediction_rows = 0
    for model_id, runs in runs_by_model.items():
        if len(runs) != len(split_records):
            raise ValueError(f"{model_id} does not have one run per split seed")
        for record, run in zip(split_records, runs, strict=True):
            summary = _read_json(run.run_dir / "summary.json")
            manifest = _read_json(run.run_dir / "run_manifest.json")
            if summary.get("split_id") != record.summary["split_id"]:
                raise ValueError(f"{model_id} did not reuse the controlled split artifact")
            if manifest.get("split_id") != record.summary["split_id"]:
                raise ValueError(f"{model_id} run manifest has a different split")
            expected = _expected_oof_identity(record.assignments_path)
            rows = read_jsonl(run.run_dir / "oof_predictions.jsonl")
            observed = {
                str(row["subject_id"]): (int(row["label"]), int(row["outer_fold"]))
                for row in rows
            }
            if len(observed) != len(rows) or observed != expected:
                raise ValueError(f"{model_id} OOF subjects/labels/folds are misaligned")
            prediction_rows += len(rows)
    return {
        "split_ids": expected_split_ids,
        "models": len(runs_by_model),
        "repetitions_per_model": len(split_records),
        "outer_folds_per_repetition": len(
            {fold for _, fold in _expected_oof_identity(split_records[0].assignments_path).values()}
        ),
        "oof_subjects_per_repetition": len(
            _expected_oof_identity(split_records[0].assignments_path)
        ),
        "oof_prediction_rows_checked": prediction_rows,
        "exact_split_subject_label_fold_alignment": True,
    }


def _exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    observed = abs(fmean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(fmean(sign * value for sign, value in zip(signs, differences, strict=True)))
        extreme += permuted >= observed - 1e-15
        total += 1
    return extreme / total


def _holm_adjust(raw_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def matched_model_comparisons(
    model_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare full minus each baseline across the same eight split repetitions."""
    full_id = "hierarchical_attention_full"
    baselines = (
        "tfidf_linear_svm",
        "chinese_roberta_mean_logistic",
        "qwen_vl_mean_mlp",
    )
    if set(model_summaries) != {full_id, *baselines}:
        raise ValueError("Matched comparisons require exactly the four controlled models")
    full_rows = model_summaries[full_id]["per_repetition"]
    full_seeds = [int(row["split_seed"]) for row in full_rows]
    comparisons: dict[str, dict[str, Any]] = {baseline: {} for baseline in baselines}

    for metric in CONTROLLED_METRICS:
        raw_p: dict[str, float] = {}
        metric_differences: dict[str, list[float]] = {}
        for baseline in baselines:
            baseline_rows = model_summaries[baseline]["per_repetition"]
            if [int(row["split_seed"]) for row in baseline_rows] != full_seeds:
                raise ValueError("Model summaries are not ordered on the same split seeds")
            if [row["split_id"] for row in baseline_rows] != [
                row["split_id"] for row in full_rows
            ]:
                raise ValueError("Model summaries do not reference the same split IDs")
            differences = [
                float(full["oof_metrics"][metric])
                - float(base["oof_metrics"][metric])
                for full, base in zip(full_rows, baseline_rows, strict=True)
            ]
            metric_differences[baseline] = differences
            raw_p[baseline] = _exact_sign_flip_pvalue(differences)
        adjusted = _holm_adjust(raw_p)
        for baseline in baselines:
            differences = metric_differences[baseline]
            interval = asdict(mean_sd_ci95(differences))
            sample_sd = stdev(differences)
            comparisons[baseline][metric] = {
                "direction": "hierarchical_attention_full minus baseline",
                "differences_by_split_seed": dict(
                    zip((str(seed) for seed in full_seeds), differences, strict=True)
                ),
                "summary": interval,
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
                "paired_standardized_mean_difference": (
                    fmean(differences) / sample_sd if sample_sd > 0 else None
                ),
                "exact_two_sided_sign_flip_p": raw_p[baseline],
                "holm_adjusted_p_within_metric": adjusted[baseline],
            }
    return {
        "comparison_unit": "matched split-seed repetition-level OOF metric",
        "contrasts": "hierarchical_attention_full minus each baseline",
        "test": "exact two-sided sign-flip randomization over 2^8 sign assignments",
        "multiplicity": "Holm adjustment across the three planned baselines within each metric",
        "interpretation_boundary": (
            "These paired summaries quantify partition sensitivity on this fixed cohort; "
            "the eight repetitions reuse subjects and are not population replicates."
        ),
        "by_baseline": comparisons,
    }


def _summary_paths(
    *, output_root: Path, prefix: str, runs: Sequence[Any], seeds: Sequence[int], kind: str
) -> tuple[Path, dict[str, Any]]:
    if kind == "tfidf":
        return summarize_repeated_tfidf_linear_svm_runs(
            runs, expected_split_seeds=seeds, output_root=output_root
        )
    if kind == "full":
        return summarize_repeated_cv_runs(
            runs, expected_split_seeds=seeds, output_root=output_root
        )
    experiment_id = (
        "frozen_chinese_roberta_mean_pool_logistic_regression"
        if kind == "chinese"
        else QWEN_VL_MEAN_MLP_EXPERIMENT_ID
    )
    boundary = (
        "Intervals quantify partition sensitivity on this fixed cohort and fixed model "
        "configuration. Repetitions reuse subjects and are not population replicates."
    )
    return summarize_repeated_oof_runs(
        runs,
        expected_split_seeds=seeds,
        output_root=output_root,
        expected_experiment_id=experiment_id,
        repeated_cv_id_prefix=prefix,
        interpretation_boundary=boundary,
    )


def run_controlled_benchmark(
    *,
    dataset: PreparedDataset,
    qwen_feature_set: PreparedFeatureSet,
    chinese_feature_set: PreparedChineseTransformerFeatureSet,
    study_config: StudyConfig,
    benchmark_config: ControlledBenchmarkConfig,
    full_config: FullExperimentConfig,
    qwen_config: QwenVLMeanMLPConfig,
    chinese_config: ChineseTransformerBaselineConfig,
    tfidf_config: TfidfLinearSVMConfig | None = None,
    on_run_complete: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> ControlledBenchmarkResult:
    required_cublas_workspace = ":4096:8"
    configured_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_cublas_workspace not in {None, required_cublas_workspace}:
        raise ValueError(
            "Controlled benchmark requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    # This occurs before any run manifest probes CUDA and before neural training.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_cublas_workspace
    deterministic_runtime = {
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": required_cublas_workspace,
        "flash_sdp": False,
        "memory_efficient_sdp": False,
        "math_sdp": True,
    }
    tfidf_config = tfidf_config or TfidfLinearSVMConfig()
    _validate_protocol(
        benchmark_config=benchmark_config,
        dataset=dataset,
        study_config=study_config,
        qwen_feature_set=qwen_feature_set,
        chinese_feature_set=chinese_feature_set,
        full_config=full_config,
        qwen_config=qwen_config,
        chinese_config=chinese_config,
        tfidf_config=tfidf_config,
    )
    input_audit = audit_controlled_inputs(
        dataset=dataset,
        study_config=study_config,
        benchmark_config=benchmark_config,
        qwen_feature_set=qwen_feature_set,
        chinese_feature_set=chinese_feature_set,
    )
    split_records = create_controlled_splits(
        dataset=dataset,
        study_config=study_config,
        benchmark_config=benchmark_config,
    )
    repository = Path(__file__).resolve().parents[3]
    source = git_state(repository)
    identity = {
        "protocol": benchmark_config.public_dict(),
        "study_config_digest": study_config.digest(),
        "dataset_id": dataset.dataset_id,
        "qwen_feature_id": qwen_feature_set.feature_id,
        "chinese_transformer_feature_id": chinese_feature_set.feature_id,
        "split_ids": [record.summary["split_id"] for record in split_records],
        "model_configurations": {
            "tfidf_linear_svm": tfidf_config.public_dict(),
            "chinese_roberta_mean_logistic": chinese_config.public_dict(),
            "qwen_vl_mean_mlp": qwen_config.public_dict(),
            "hierarchical_attention_full": full_config.public_dict(),
        },
        "deterministic_runtime": deterministic_runtime,
        "source_code": {"commit": source["commit"], "dirty": source["dirty"]},
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    benchmark_id = f"controlled-benchmark-{digest}"
    benchmark_dir = study_config.artifact_root / "controlled_benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / "summary.json"
    if summary_path.is_file():
        return ControlledBenchmarkResult(benchmark_dir, _read_json(summary_path))
    write_json_atomic(benchmark_dir / "identity.json", {"schema_version": 1, **identity})
    write_json_atomic(benchmark_dir / "input_audit.json", input_audit)

    seeds = benchmark_config.splits.seeds
    repeated_root = study_config.artifact_root / "repeated_cv"
    runs_by_model: dict[str, list[Any]] = {
        model_id: [] for model_id in benchmark_config.models
    }
    completed: list[dict[str, Any]] = []

    def record(model_id: str, split_seed: int, result: Any) -> None:
        completed.append(
            {
                "model_id": model_id,
                "split_seed": split_seed,
                "run_id": result.summary["run_id"],
                "oof_metrics": result.summary["oof_metrics"],
            }
        )
        write_json_atomic(
            benchmark_dir / "progress.json",
            {
                "schema_version": 1,
                "benchmark_id": benchmark_id,
                "status": "running",
                "completed_repetition_runs": len(completed),
                "planned_repetition_runs": len(seeds) * len(runs_by_model),
                "completed": completed,
            },
        )
        if on_run_complete is not None:
            on_run_complete(model_id, split_seed, result.summary)

    for split in split_records:
        result = train_tfidf_linear_svm(
            dataset=dataset,
            split_assignments_path=split.assignments_path,
            study_config=study_config,
            output_root=study_config.run_dir,
            keyword_condition="removed",
            configuration=tfidf_config,
        )
        runs_by_model["tfidf_linear_svm"].append(result)
        record("tfidf_linear_svm", split.seed, result)

    for split in split_records:
        result = train_chinese_transformer_baseline(
            feature_set=chinese_feature_set,
            split_assignments_path=split.assignments_path,
            study_config=study_config,
            configuration=chinese_config,
            output_root=study_config.run_dir,
        )
        runs_by_model["chinese_roberta_mean_logistic"].append(result)
        record("chinese_roberta_mean_logistic", split.seed, result)

    for split in split_records:
        result = train_qwen_vl_mean_mlp(
            feature_set=qwen_feature_set,
            split_assignments_path=split.assignments_path,
            study_config=study_config,
            configuration=qwen_config,
            output_root=study_config.run_dir,
            run_context={
                "controlled_benchmark": True,
                "deterministic_runtime": deterministic_runtime,
            },
        )
        runs_by_model["qwen_vl_mean_mlp"].append(result)
        record("qwen_vl_mean_mlp", split.seed, result)

    for split in split_records:
        result = train_hierarchical_full(
            feature_set=qwen_feature_set,
            split_assignments_path=split.assignments_path,
            study_config=study_config,
            full_config=full_config,
            output_root=study_config.run_dir,
            additional_parameters={
                "run_context": {
                    "controlled_benchmark": True,
                    "deterministic_runtime": deterministic_runtime,
                }
            },
        )
        runs_by_model["hierarchical_attention_full"].append(result)
        record("hierarchical_attention_full", split.seed, result)

    repeated_specs = {
        "tfidf_linear_svm": ("repeated-tfidf", "tfidf"),
        "chinese_roberta_mean_logistic": ("repeated-chinese-transformer", "chinese"),
        "qwen_vl_mean_mlp": ("repeated-qwen-vl-mean-mlp", "qwen"),
        "hierarchical_attention_full": ("repeated-full", "full"),
    }
    repeated_dirs: dict[str, Path] = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    for model_id, (prefix, kind) in repeated_specs.items():
        repeated_dir, repeated_summary = _summary_paths(
            output_root=repeated_root,
            prefix=prefix,
            runs=runs_by_model[model_id],
            seeds=seeds,
            kind=kind,
        )
        repeated_dirs[model_id] = repeated_dir
        model_summaries[model_id] = repeated_summary

    alignment_audit = audit_model_oof_alignment(
        split_records=split_records, runs_by_model=runs_by_model
    )
    comparisons = matched_model_comparisons(model_summaries)
    summary = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "status": "complete",
        **identity,
        "material_passport": {
            "data": {
                "dataset_id": dataset.dataset_id,
                "cohort": input_audit,
                "keyword_condition": "removed",
                "maximum_posts_per_subject": (
                    study_config.dataset.post_selection.maximum_posts_per_subject
                ),
            },
            "method": {
                "split_repetitions": len(seeds),
                "outer_folds": benchmark_config.splits.outer_folds,
                "split_seeds": list(seeds),
                "training_base_seed": benchmark_config.training_base_seed,
                "fold_training_seed_rule": "training_base_seed + outer_fold",
                "classification_threshold": benchmark_config.classification_threshold,
                "primary_metric": benchmark_config.primary_metric,
            },
            "evidence": {
                "model_repeated_cv_summaries": {
                    model_id: _artifact_relative(path / "summary.json", study_config.artifact_root)
                    for model_id, path in repeated_dirs.items()
                },
                "alignment_audit": "embedded_below",
                "paired_comparisons": "embedded_below",
            },
            "provenance": {
                "git": source,
                "study_config_digest": study_config.digest(),
            },
        },
        "model_results": {
            model_id: {
                "repeated_cv_id": model_summaries[model_id]["repeated_cv_id"],
                "summary_path": _artifact_relative(
                    repeated_dirs[model_id] / "summary.json", study_config.artifact_root
                ),
                "metric_summary": model_summaries[model_id]["metric_summary"],
                "per_repetition": model_summaries[model_id]["per_repetition"],
            }
            for model_id in benchmark_config.models
        },
        "alignment_audit": alignment_audit,
        "paired_comparisons": comparisons,
        "trained_fold_models": len(seeds) * benchmark_config.splits.outer_folds * 4,
        "interpretation_boundary": (
            "This is a fixed-cohort endpoint comparison under shared data and partitions. "
            "It is not an architecture-only ablation and does not imply population-level "
            "uncertainty from eight independent samples."
        ),
    }
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        benchmark_dir / "progress.json",
        {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "status": "complete",
            "completed_repetition_runs": len(completed),
            "planned_repetition_runs": len(completed),
        },
    )
    return ControlledBenchmarkResult(benchmark_dir, summary)
