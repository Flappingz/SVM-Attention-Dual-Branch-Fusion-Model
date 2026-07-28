"""Command-line entry points for data preparation and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.data.prepare import prepare_dataset
from ocd_v3.data.raw import audit_raw_dataset
from ocd_v3.evaluation.splits import create_split_artifact
from ocd_v3.io import write_json_atomic


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_repeated_cv_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-seeds", required=True)
    parser.add_argument("--minimum-posts-per-subject", type=int, default=1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal OCD risk-signal research pipeline"
    )
    parser.add_argument("--config", required=True, help="Study configuration with local paths")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-raw", help="Audit governed raw data without copying it")
    audit.add_argument("--output", default=None)
    commands.add_parser("prepare", help="Prepare pseudonymized local data outside the repository")

    splits = commands.add_parser("make-splits", help="Create one fixed user-level split artifact")
    splits.add_argument("--dataset-dir", required=True)
    splits.add_argument("--minimum-posts-per-subject", type=int, default=1)

    validate = commands.add_parser(
        "validate-full-config", help="Validate a model configuration"
    )
    validate.add_argument("--full-config", required=True)

    features = commands.add_parser(
        "build-features", help="Build frozen Qwen3-VL feature bundles"
    )
    features.add_argument("--dataset-dir", required=True)
    features.add_argument("--full-config", required=True)
    features.add_argument(
        "--keyword-condition",
        choices=["original", "masked", "removed"],
        default="original",
    )
    features.add_argument(
        "--embedding-modalities",
        choices=["text", "image", "text,image"],
        default="text,image",
    )
    features.add_argument("--source-feature-dir", default=None)
    features.add_argument("--subject-limit", type=int, default=None)
    features.add_argument("--post-limit", type=int, default=None)

    chinese_features = commands.add_parser(
        "build-chinese-transformer-features",
        help="Build frozen Chinese-RoBERTa baseline features",
    )
    chinese_features.add_argument("--dataset-dir", required=True)
    chinese_features.add_argument("--baseline-config", required=True)
    chinese_features.add_argument("--model-path", required=True)
    chinese_features.add_argument(
        "--keyword-condition",
        choices=["original", "masked", "removed"],
        default="original",
    )
    chinese_features.add_argument("--subject-limit", type=int, default=None)
    chinese_features.add_argument("--post-limit", type=int, default=None)

    full = commands.add_parser("train-full", help="Run one hierarchical auxiliary-branch CV")
    full.add_argument("--feature-dir", required=True)
    full.add_argument("--split-assignments", required=True)
    full.add_argument("--full-config", required=True)
    full.add_argument("--folds", default=None)
    full.add_argument("--epoch-limit", type=int, default=None)
    full.add_argument("--exclude-metadata", action="store_true")

    full_repeated = commands.add_parser(
        "train-full-repeated-cv", help="Run repeated CV for the hierarchical auxiliary branch"
    )
    _add_repeated_cv_arguments(full_repeated)
    full_repeated.add_argument("--feature-dir", required=True)
    full_repeated.add_argument("--full-config", required=True)
    full_repeated.add_argument("--exclude-metadata", action="store_true")

    tfidf_repeated = commands.add_parser(
        "train-tfidf-linear-svm-repeated-cv", help="Run the TF-IDF baseline"
    )
    _add_repeated_cv_arguments(tfidf_repeated)
    tfidf_repeated.add_argument(
        "--keyword-condition",
        choices=["original", "masked", "removed"],
        default="original",
    )

    qwen_mean_repeated = commands.add_parser(
        "train-qwen-vl-mean-mlp-repeated-cv", help="Run the Qwen mean-MLP baseline"
    )
    _add_repeated_cv_arguments(qwen_mean_repeated)
    qwen_mean_repeated.add_argument("--feature-dir", required=True)
    qwen_mean_repeated.add_argument("--baseline-config", required=True)

    chinese_repeated = commands.add_parser(
        "train-chinese-transformer-repeated-cv", help="Run the RoBERTa mean-LR baseline"
    )
    _add_repeated_cv_arguments(chinese_repeated)
    chinese_repeated.add_argument("--feature-dir", required=True)
    chinese_repeated.add_argument("--baseline-config", required=True)

    benchmark = commands.add_parser(
        "run-controlled-benchmark", help="Run the paper's shared-cohort four-model benchmark"
    )
    benchmark.add_argument("--benchmark-config", required=True)
    benchmark.add_argument("--dataset-dir", required=True)
    benchmark.add_argument("--qwen-feature-dir", required=True)
    benchmark.add_argument("--chinese-transformer-feature-dir", required=True)
    benchmark.add_argument("--full-config", required=True)
    benchmark.add_argument("--qwen-baseline-config", required=True)
    benchmark.add_argument("--chinese-transformer-baseline-config", required=True)
    return parser


def _parse_folds(value: str | None) -> list[int] | None:
    if value is None:
        return None
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--folds must include at least one integer")
    return result


def _progress(split_seed: int, split_summary: dict[str, object], result: object) -> None:
    _emit(
        {
            "completed_split_seed": split_seed,
            "split_id": split_summary["split_id"],
            "run_id": result.summary["run_id"],
            "oof_metrics": result.summary["oof_metrics"],
        }
    )


def _emit_repeated(result: object) -> None:
    _emit(
        {
            "repeated_cv_dir": str(result.repeated_cv_dir),
            "metric_summary": result.summary["metric_summary"],
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "audit-raw":
        report = audit_raw_dataset(config)
        output = Path(args.output) if args.output else config.audit_dir / "raw_audit.json"
        write_json_atomic(output, report.to_dict())
        _emit(
            {
                "raw_inventory_sha256": report.raw_inventory_sha256,
                "groups": [
                    {
                        "label_name": group.label_name,
                        "listed_users": group.unique_listed_users,
                        "post_rows": group.post_rows,
                    }
                    for group in report.groups
                ],
                "report": str(output),
            }
        )
        return

    if args.command == "prepare":
        dataset_dir, manifest = prepare_dataset(config)
        _emit(
            {
                "dataset_id": manifest["dataset_id"],
                "groups": manifest["groups"],
                "dataset_dir": str(dataset_dir),
                "identifier_policy": manifest["identifier_policy"],
            }
        )
        return

    if args.command == "make-splits":
        split_path, summary = create_split_artifact(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            fold_count=config.evaluation.outer_folds,
            validation_fraction=config.evaluation.validation_fraction_within_outer_train,
            seed=config.evaluation.split_seed,
            output_root=config.split_dir,
            minimum_posts_per_subject=args.minimum_posts_per_subject,
        )
        _emit(
            {
                "split_id": summary["split_id"],
                "assignments": str(split_path),
                "summary": summary["summary"],
                "cohort": summary["cohort"],
            }
        )
        return

    if args.command == "validate-full-config":
        from ocd_v3.experiments.full_config import load_full_experiment_config

        full_config = load_full_experiment_config(args.full_config)
        _emit(
            {
                "model_id": full_config.encoder.model_id,
                "revision": full_config.encoder.revision,
                "representations": list(full_config.encoder.representations),
                "architecture_variant": full_config.model.architecture_variant,
                "maximum_posts_per_subject": (
                    config.dataset.post_selection.maximum_posts_per_subject
                ),
            }
        )
        return

    if args.command == "build-features":
        from ocd_v3.experiments.full_config import load_full_experiment_config
        from ocd_v3.features.build import build_feature_set
        from ocd_v3.features.training_data import PreparedFeatureSet

        result = build_feature_set(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            study_config=config,
            full_config=load_full_experiment_config(args.full_config),
            keyword_condition=args.keyword_condition,
            output_root=config.feature_dir,
            subject_limit=args.subject_limit,
            post_limit=args.post_limit,
            source_feature_set=(
                PreparedFeatureSet(Path(args.source_feature_dir))
                if args.source_feature_dir
                else None
            ),
            embedding_input_modalities=tuple(args.embedding_modalities.split(",")),
        )
        _emit(
            {
                "feature_id": result.manifest["feature_id"],
                "feature_dir": str(result.feature_dir),
                "complete_dataset": result.manifest["complete_dataset"],
                "counts": result.manifest["counts"],
            }
        )
        return

    if args.command == "build-chinese-transformer-features":
        from ocd_v3.experiments.chinese_transformer_config import load_chinese_transformer_config
        from ocd_v3.features.chinese_transformer import build_chinese_transformer_features

        baseline_config = load_chinese_transformer_config(args.baseline_config)
        result = build_chinese_transformer_features(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            study_config=config,
            configuration=baseline_config.encoder,
            model_path=Path(args.model_path),
            output_root=config.feature_dir,
            keyword_condition=args.keyword_condition,
            subject_limit=args.subject_limit,
            post_limit=args.post_limit,
        )
        _emit(
            {
                "feature_id": result.manifest["feature_id"],
                "feature_dir": str(result.feature_dir),
                "complete_dataset": result.manifest["complete_dataset"],
                "counts": result.manifest["counts"],
            }
        )
        return

    if args.command == "train-full":
        from ocd_v3.experiments.full_config import load_full_experiment_config
        from ocd_v3.experiments.train_full import train_hierarchical_full
        from ocd_v3.features.training_data import PreparedFeatureSet

        result = train_hierarchical_full(
            feature_set=PreparedFeatureSet(Path(args.feature_dir)),
            split_assignments_path=Path(args.split_assignments),
            study_config=config,
            full_config=load_full_experiment_config(args.full_config),
            output_root=config.run_dir,
            selected_folds=_parse_folds(args.folds),
            epoch_limit=args.epoch_limit,
            include_metadata=not args.exclude_metadata,
        )
        _emit(
            {
                "run_id": result.summary["run_id"],
                "run_dir": str(result.run_dir),
                "oof_metrics": result.summary.get("oof_metrics"),
            }
        )
        return

    from ocd_v3.experiments.seed_sweep import parse_seed_spec

    if args.command == "train-full-repeated-cv":
        from ocd_v3.experiments.full_config import load_full_experiment_config
        from ocd_v3.experiments.repeated_cv import run_repeated_full_cv
        from ocd_v3.features.training_data import PreparedFeatureSet

        result = run_repeated_full_cv(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            feature_set=PreparedFeatureSet(Path(args.feature_dir)),
            study_config=config,
            full_config=load_full_experiment_config(args.full_config),
            run_output_root=config.run_dir,
            split_output_root=config.split_dir,
            repeated_cv_output_root=config.artifact_root / "repeated_cv",
            split_seeds=parse_seed_spec(args.split_seeds),
            minimum_posts_per_subject=args.minimum_posts_per_subject,
            include_metadata=not args.exclude_metadata,
            on_run_complete=_progress,
        )
        _emit_repeated(result)
        return

    if args.command == "train-tfidf-linear-svm-repeated-cv":
        from ocd_v3.experiments.tfidf_repeated_cv import run_repeated_tfidf_linear_svm_cv

        result = run_repeated_tfidf_linear_svm_cv(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            study_config=config,
            run_output_root=config.run_dir,
            split_output_root=config.split_dir,
            repeated_cv_output_root=config.artifact_root / "repeated_cv",
            split_seeds=parse_seed_spec(args.split_seeds),
            keyword_condition=args.keyword_condition,
            minimum_posts_per_subject=args.minimum_posts_per_subject,
            on_run_complete=_progress,
        )
        _emit_repeated(result)
        return

    if args.command == "train-qwen-vl-mean-mlp-repeated-cv":
        from ocd_v3.experiments.qwen_vl_mean_mlp_config import load_qwen_vl_mean_mlp_config
        from ocd_v3.experiments.qwen_vl_mean_mlp_repeated_cv import run_repeated_qwen_vl_mean_mlp_cv
        from ocd_v3.features.training_data import PreparedFeatureSet

        result = run_repeated_qwen_vl_mean_mlp_cv(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            feature_set=PreparedFeatureSet(Path(args.feature_dir)),
            study_config=config,
            configuration=load_qwen_vl_mean_mlp_config(args.baseline_config),
            run_output_root=config.run_dir,
            split_output_root=config.split_dir,
            repeated_cv_output_root=config.artifact_root / "repeated_cv",
            split_seeds=parse_seed_spec(args.split_seeds),
            minimum_posts_per_subject=args.minimum_posts_per_subject,
            on_run_complete=_progress,
        )
        _emit_repeated(result)
        return

    if args.command == "train-chinese-transformer-repeated-cv":
        from ocd_v3.experiments.chinese_transformer_config import load_chinese_transformer_config
        from ocd_v3.experiments.chinese_transformer_repeated_cv import (
            run_repeated_chinese_transformer_cv,
        )
        from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet

        result = run_repeated_chinese_transformer_cv(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            feature_set=PreparedChineseTransformerFeatureSet(Path(args.feature_dir)),
            study_config=config,
            configuration=load_chinese_transformer_config(args.baseline_config),
            run_output_root=config.run_dir,
            split_output_root=config.split_dir,
            repeated_cv_output_root=config.artifact_root / "repeated_cv",
            split_seeds=parse_seed_spec(args.split_seeds),
            minimum_posts_per_subject=args.minimum_posts_per_subject,
            on_run_complete=_progress,
        )
        _emit_repeated(result)
        return

    if args.command == "run-controlled-benchmark":
        from ocd_v3.experiments.chinese_transformer_config import load_chinese_transformer_config
        from ocd_v3.experiments.controlled_benchmark import run_controlled_benchmark
        from ocd_v3.experiments.controlled_benchmark_config import load_controlled_benchmark_config
        from ocd_v3.experiments.full_config import load_full_experiment_config
        from ocd_v3.experiments.qwen_vl_mean_mlp_config import load_qwen_vl_mean_mlp_config
        from ocd_v3.features.chinese_transformer import PreparedChineseTransformerFeatureSet
        from ocd_v3.features.training_data import PreparedFeatureSet

        result = run_controlled_benchmark(
            dataset=PreparedDataset(Path(args.dataset_dir)),
            qwen_feature_set=PreparedFeatureSet(Path(args.qwen_feature_dir)),
            chinese_feature_set=PreparedChineseTransformerFeatureSet(Path(args.chinese_transformer_feature_dir)),
            study_config=config,
            benchmark_config=load_controlled_benchmark_config(args.benchmark_config),
            full_config=load_full_experiment_config(args.full_config),
            qwen_config=load_qwen_vl_mean_mlp_config(args.qwen_baseline_config),
            chinese_config=load_chinese_transformer_config(args.chinese_transformer_baseline_config),
        )
        _emit(
            {
                "benchmark_id": result.summary["benchmark_id"],
                "benchmark_dir": str(result.benchmark_dir),
                "status": result.summary["status"],
                "model_results": result.summary["model_results"],
            }
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
