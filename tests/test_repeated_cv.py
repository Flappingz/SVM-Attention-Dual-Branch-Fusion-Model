from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocd_v3.experiments.repeated_cv import summarize_repeated_cv_runs
from ocd_v3.experiments.train_full import TrainFullResult


def _write_repeated_run(
    root: Path,
    *,
    split_seed: int,
    fold_assignments: tuple[int, ...],
    f1: float,
) -> TrainFullResult:
    run_id = f"run-split-seed-{split_seed}"
    split_id = f"split-seed-{split_seed}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "dataset_id": "dataset-test",
        "split_id": split_id,
        "source_code": {"commit": "abc", "dirty": False},
        "created_at": f"time-{split_seed}",
        "seeds": {"split": split_seed, "training_base": 101},
        "parameters": {
            "feature_id": "feature-test",
            "keyword_condition": "original",
            "classification_threshold": 0.5,
            "full_experiment": {"training": {"seed": 101, "epochs": 1}},
        },
    }
    predictions = [
        {
            "subject_id": f"sub-{index}",
            "outer_fold": fold,
            "label": index % 2,
            "score": 0.8 if index % 2 else 0.2,
        }
        for index, fold in enumerate(fold_assignments)
    ]
    metrics = {
        "n": len(predictions),
        "tp": len(predictions) // 2,
        "tn": len(predictions) // 2,
        "fp": 0,
        "fn": 0,
        "accuracy": f1,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "roc_auc": f1,
    }
    summary = {
        "run_id": run_id,
        "split_id": split_id,
        "complete_outer_cv": True,
        "oof_metrics": metrics,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "oof_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    return TrainFullResult(run_dir=run_dir, summary=summary)


class RepeatedCVTests(unittest.TestCase):
    def test_summarizes_distinct_split_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                _write_repeated_run(
                    root,
                    split_seed=42,
                    fold_assignments=(0, 0, 1, 1, 2, 2, 3, 3, 4, 4),
                    f1=0.6,
                ),
                _write_repeated_run(
                    root,
                    split_seed=43,
                    fold_assignments=(0, 1, 0, 1, 2, 3, 2, 4, 3, 4),
                    f1=0.8,
                ),
            ]
            output_dir, summary = summarize_repeated_cv_runs(
                runs,
                expected_split_seeds=[42, 43],
                output_root=root / "repeated_cv",
            )
            self.assertEqual(summary["n_split_repetitions"], 2)
            self.assertEqual(summary["outer_folds_per_repetition"], 5)
            self.assertEqual(summary["trained_fold_models"], 10)
            self.assertEqual(summary["oof_subjects_per_repetition"], 10)
            self.assertAlmostEqual(summary["metric_summary"]["f1"]["mean"], 0.7)
            self.assertEqual(summary["training_base_seed"], 101)
            self.assertNotEqual(
                summary["per_repetition"][0]["outer_partition_sha256"],
                summary["per_repetition"][1]["outer_partition_sha256"],
            )
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_rejects_duplicate_outer_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignments = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
            runs = [
                _write_repeated_run(
                    root, split_seed=42, fold_assignments=assignments, f1=0.6
                ),
                _write_repeated_run(
                    root, split_seed=43, fold_assignments=assignments, f1=0.8
                ),
            ]
            with self.assertRaisesRegex(ValueError, "same outer partition"):
                summarize_repeated_cv_runs(
                    runs,
                    expected_split_seeds=[42, 43],
                    output_root=root / "repeated_cv",
                )


if __name__ == "__main__":
    unittest.main()
