"""Run the Qwen sequence-moments SVM repeated-CV experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.experiments.qwen_sequence_svm import (
    DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE,
    QwenSequenceSVMResult,
)
from ocd_v3.experiments.qwen_sequence_svm_config import (
    load_qwen_sequence_svm_config,
)
from ocd_v3.experiments.qwen_sequence_svm_repeated_cv import (
    run_repeated_qwen_sequence_svm_cv,
)
from ocd_v3.experiments.seed_sweep import parse_seed_spec
from ocd_v3.features.training_data import PreparedFeatureSet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--split-seeds", required=True)
    parser.add_argument("--minimum-posts-per-subject", type=int, default=1)
    parser.add_argument(
        "--selection-provenance",
        default=DEFAULT_QWEN_SEQUENCE_SELECTION_PROVENANCE,
        help=(
            "Human-readable description of when the model family was finalized. "
            "This is recorded for transparency and is not used by the estimator."
        ),
    )
    parser.add_argument("--interpretation-boundary")
    args = parser.parse_args()

    study = load_config(args.study_config)
    artifact_root = study.artifact_root

    def report(
        seed: int,
        split_summary: dict[str, object],
        result: QwenSequenceSVMResult,
    ) -> None:
        run_summary = result.summary
        print(
            json.dumps(
                {
                    "split_seed": seed,
                    "split_id": split_summary["split_id"],
                    "run_id": run_summary["run_id"],
                    "oof_metrics": run_summary["oof_metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = run_repeated_qwen_sequence_svm_cv(
        dataset=PreparedDataset(args.dataset_dir),
        feature_set=PreparedFeatureSet(args.feature_dir),
        study_config=study,
        configuration=load_qwen_sequence_svm_config(args.model_config),
        run_output_root=artifact_root / "runs",
        split_output_root=artifact_root / "splits",
        repeated_cv_output_root=artifact_root / "repeated_cv",
        split_seeds=parse_seed_spec(args.split_seeds),
        minimum_posts_per_subject=args.minimum_posts_per_subject,
        selection_provenance=args.selection_provenance,
        interpretation_boundary=args.interpretation_boundary,
        on_run_complete=report,
    )
    print(
        json.dumps(
            {
                "repeated_cv_dir": str(result.repeated_cv_dir),
                "repeated_cv_id": result.summary["repeated_cv_id"],
                "metric_summary": result.summary["metric_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
