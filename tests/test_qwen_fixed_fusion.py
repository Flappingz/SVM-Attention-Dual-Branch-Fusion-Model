from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocd_v3.experiments.qwen_fixed_fusion import (
    build_fixed_qwen_metadata_fusion,
    build_fixed_qwen_sequence_fusion,
)
from ocd_v3.io import read_jsonl, write_json_atomic, write_jsonl_atomic


class QwenFixedFusionTests(unittest.TestCase):
    def test_builds_aligned_pure_qwen_fusion_without_tfidf_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            primary_repetitions: list[dict[str, object]] = []
            secondary_repetitions: list[dict[str, object]] = []
            split_ids: list[str] = []
            for seed in (58, 59):
                split_id = f"split-{seed}"
                primary_run = f"primary-{seed}"
                secondary_run = f"secondary-{seed}"
                split_ids.append(split_id)
                primary_repetitions.append(
                    {"split_seed": seed, "split_id": split_id, "run_id": primary_run}
                )
                secondary_repetitions.append(
                    {"split_seed": seed, "split_id": split_id, "run_id": secondary_run}
                )
                rows = [
                    {
                        "subject_id": "control",
                        "outer_fold": 0,
                        "label": 0,
                        "score": 0.2 + (seed - 58) * 0.01,
                    },
                    {
                        "subject_id": "case",
                        "outer_fold": 1,
                        "label": 1,
                        "score": 0.8 - (seed - 58) * 0.01,
                    },
                ]
                write_jsonl_atomic(
                    artifact_root / "runs" / primary_run / "oof_predictions.jsonl",
                    rows,
                )
                secondary_rows = [
                    {**rows[0], "score": 0.4},
                    {**rows[1], "score": 0.6},
                ]
                write_jsonl_atomic(
                    artifact_root / "runs" / secondary_run / "oof_predictions.jsonl",
                    secondary_rows,
                )
                write_json_atomic(
                    artifact_root / "runs" / secondary_run / "run_manifest.json",
                    {
                        "parameters": {
                            "full_experiment": {
                                "encoder": {
                                    "model_id": "Qwen/Qwen3-VL-Embedding-2B"
                                }
                            }
                        }
                    },
                )
            primary_summary = root / "primary.json"
            secondary_summary = root / "secondary.json"
            write_json_atomic(
                primary_summary,
                {
                    "dataset_id": "dataset-test",
                    "repeated_cv_id": "primary-repeated",
                    "split_ids": split_ids,
                    "parameters": {
                        "feature_id": "features-test",
                        "keyword_condition": "removed",
                        "configuration": {
                            "encoder": {
                                "model_id": "Qwen/Qwen3-VL-Embedding-2B"
                            }
                        },
                    },
                    "per_repetition": primary_repetitions,
                },
            )
            write_json_atomic(
                secondary_summary,
                {
                    "dataset_id": "dataset-test",
                    "feature_id": "features-test",
                    "keyword_condition": "removed",
                    "repeated_cv_id": "secondary-repeated",
                    "split_ids": split_ids,
                    "per_repetition": secondary_repetitions,
                },
            )

            output_dir, summary = build_fixed_qwen_sequence_fusion(
                artifact_root=artifact_root,
                primary_summary_path=primary_summary,
                secondary_summary_path=secondary_summary,
                output_root=root / "fusions",
                repository=root,
                secondary_weight=0.1,
                threshold=0.5,
                selection_provenance=(
                    "PRE_SPECIFIED_PARTITION_CONFIRMATION: synthetic no TF-IDF input"
                ),
            )
            self.assertEqual(summary["primary_weight"], 0.9)
            self.assertEqual(summary["secondary_weight"], 0.1)
            self.assertEqual(summary["metric_summary"]["f1"]["n"], 2)
            rows = read_jsonl(output_dir / "oof_predictions.jsonl")
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(rows[0]["score"], 0.9 * 0.8 + 0.1 * 0.6)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["experiment_id"],
                "fixed_pure_qwen_sequence_score_fusion",
            )
            self.assertNotIn("tfidf_repeated_cv_id", manifest)

            primary_payload = json.loads(primary_summary.read_text(encoding="utf-8"))
            primary_payload["parameters"]["keyword_condition"] = "original"
            write_json_atomic(primary_summary, primary_payload)
            secondary_payload = json.loads(
                secondary_summary.read_text(encoding="utf-8")
            )
            secondary_payload["keyword_condition"] = "original"
            write_json_atomic(secondary_summary, secondary_payload)
            original_dir, original_summary = build_fixed_qwen_sequence_fusion(
                artifact_root=artifact_root,
                primary_summary_path=primary_summary,
                secondary_summary_path=secondary_summary,
                output_root=root / "original-fusions",
                repository=root,
                secondary_weight=0.1,
                threshold=0.5,
                selection_provenance="LOCKED_ORIGINAL_SYNTHETIC_TEST",
            )
            self.assertEqual(original_summary["keyword_condition"], "original")
            original_manifest = json.loads(
                (original_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("Keyword-post-original", original_manifest["input_constraint"])

            primary_payload["parameters"]["keyword_condition"] = "removed"
            write_json_atomic(primary_summary, primary_payload)
            secondary_payload["keyword_condition"] = "removed"
            write_json_atomic(secondary_summary, secondary_payload)

            for row in secondary_repetitions:
                write_json_atomic(
                    artifact_root
                    / "runs"
                    / str(row["run_id"])
                    / "run_manifest.json",
                    {
                        "experiment_id": "metadata_only_mean_logistic_regression",
                        "parameters": {
                            "baseline": "metadata_only_logistic_regression",
                            "feature_id": "features-metadata-test",
                            "keyword_condition": "removed",
                            "input_features": ["log1p_likes", "posting_time_sin"],
                        },
                    },
                )
            metadata_dir, metadata_summary = build_fixed_qwen_metadata_fusion(
                artifact_root=artifact_root,
                primary_summary_path=primary_summary,
                secondary_summary_path=secondary_summary,
                output_root=root / "metadata-fusions",
                repository=root,
                secondary_weight=0.2,
                threshold=0.5,
                selection_provenance="EXPLORATORY_POST_OOF_SELECTION: synthetic",
            )
            self.assertEqual(
                metadata_summary["experiment_id"],
                "fixed_qwen_sequence_metadata_score_fusion",
            )
            self.assertEqual(metadata_summary["secondary_kind"], "metadata")
            self.assertEqual(metadata_summary["secondary_weight"], 0.2)
            metadata_manifest = json.loads(
                (metadata_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("interaction counts", metadata_manifest["input_constraint"])


if __name__ == "__main__":
    unittest.main()
