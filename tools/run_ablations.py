"""Run the fixed modality and architecture ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.experiments.ablation_report import build_ablation_report
from ocd_v3.experiments.full_config import load_full_experiment_config
from ocd_v3.experiments.metadata_logistic import run_repeated_metadata_logistic_cv
from ocd_v3.experiments.repeated_cv import run_repeated_full_cv
from ocd_v3.experiments.seed_sweep import parse_seed_spec
from ocd_v3.features.training_data import PreparedFeatureSet


def _progress(condition: str) -> Any:
    def report(split_seed: int, split_summary: dict[str, Any], result: Any) -> None:
        print(
            json.dumps(
                {
                    "condition": condition,
                    "completed_split_seed": split_seed,
                    "split_id": split_summary["split_id"],
                    "run_id": result.summary["run_id"],
                    "oof_metrics": result.summary["oof_metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--multimodal-feature-dir", required=True)
    parser.add_argument("--text-feature-dir", required=True)
    parser.add_argument("--image-feature-dir", required=True)
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--no-post-config", required=True)
    parser.add_argument("--no-user-config", required=True)
    parser.add_argument("--mean-both-config", required=True)
    parser.add_argument("--split-seeds", default="58-65")
    parser.add_argument("--minimum-posts-per-subject", type=int, default=21)
    parser.add_argument("--full-reference-summary", required=True)
    parser.add_argument("--report-output-dir", required=True)
    args = parser.parse_args()

    study = load_config(args.study_config)
    dataset = PreparedDataset(Path(args.dataset_dir))
    features = {
        "multimodal": PreparedFeatureSet(Path(args.multimodal_feature_dir)),
        "text": PreparedFeatureSet(Path(args.text_feature_dir)),
        "image": PreparedFeatureSet(Path(args.image_feature_dir)),
    }
    configs = {
        "full": load_full_experiment_config(args.full_config),
        "without_post_cross_attention": load_full_experiment_config(
            args.no_post_config
        ),
        "without_user_self_attention": load_full_experiment_config(
            args.no_user_config
        ),
        "mean_pooling_at_both_levels": load_full_experiment_config(
            args.mean_both_config
        ),
    }
    seeds = parse_seed_spec(args.split_seeds)
    common = {
        "dataset": dataset,
        "study_config": study,
        "run_output_root": study.run_dir,
        "split_output_root": study.split_dir,
        "repeated_cv_output_root": study.artifact_root / "repeated_cv",
        "split_seeds": seeds,
        "minimum_posts_per_subject": args.minimum_posts_per_subject,
    }

    metadata = run_repeated_metadata_logistic_cv(
        feature_set=features["multimodal"],
        on_run_complete=_progress("metadata"),
        **common,
    )

    neural_specs = (
        ("text", "text", "full", False),
        ("image", "image", "full", False),
        ("text+image", "multimodal", "full", False),
        ("text+metadata", "text", "full", True),
        (
            "without_post_cross_attention",
            "multimodal",
            "without_post_cross_attention",
            True,
        ),
        (
            "without_user_self_attention",
            "multimodal",
            "without_user_self_attention",
            True,
        ),
        (
            "mean_pooling_at_both_levels",
            "multimodal",
            "mean_pooling_at_both_levels",
            True,
        ),
    )
    completed: dict[str, Any] = {}
    for condition, feature_name, config_name, include_metadata in neural_specs:
        completed[condition] = run_repeated_full_cv(
            feature_set=features[feature_name],
            full_config=configs[config_name],
            include_metadata=include_metadata,
            on_run_complete=_progress(condition),
            **common,
        )

    full_reference = Path(args.full_reference_summary).resolve()
    modality_paths = {
        "text": completed["text"].repeated_cv_dir / "summary.json",
        "image": completed["image"].repeated_cv_dir / "summary.json",
        "metadata": metadata.repeated_cv_dir / "summary.json",
        "text+image": completed["text+image"].repeated_cv_dir / "summary.json",
        "text+metadata": completed["text+metadata"].repeated_cv_dir / "summary.json",
        "text+image+metadata": full_reference,
    }
    architecture_paths = {
        "full": full_reference,
        "without_post_cross_attention": (
            completed["without_post_cross_attention"].repeated_cv_dir / "summary.json"
        ),
        "without_user_self_attention": (
            completed["without_user_self_attention"].repeated_cv_dir / "summary.json"
        ),
        "mean_pooling_at_both_levels": (
            completed["mean_pooling_at_both_levels"].repeated_cv_dir / "summary.json"
        ),
    }
    output_path, report = build_ablation_report(
        modality_summary_paths=modality_paths,
        architecture_summary_paths=architecture_paths,
        output_dir=Path(args.report_output_dir),
    )
    print(
        json.dumps(
            {
                "report": str(output_path),
                "protocol": report["protocol"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
