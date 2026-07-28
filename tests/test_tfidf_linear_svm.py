from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import write_study_config

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.experiments.tfidf_linear_svm import (
    TfidfLinearSVMConfig,
    train_tfidf_linear_svm,
)
from ocd_v3.experiments.tfidf_repeated_cv import run_repeated_tfidf_linear_svm_cv
from ocd_v3.io import write_jsonl_atomic


def _write_synthetic_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset"
    posts_dir = dataset_dir / "posts"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "dataset-text-test",
                "post_selection_for_models": {
                    "maximum_posts_per_subject": 2,
                    "strategy": "most_recent",
                },
            }
        ),
        encoding="utf-8",
    )
    subject_rows: list[dict[str, object]] = []
    for index in range(10):
        subject_id = f"subject-{index:02d}"
        label = index % 2
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label_name": "self_reported_ocd" if label else "control",
                "label_id": label,
                "post_count": 2,
                "media_count": 0,
                "eligible": True,
            }
        )
        # The snowman only appears in fold-0 test documents.  It must not be
        # learned by the fold-0 train-only TF-IDF vocabulary.
        text = (
            ("阳性特征" if label else "阴性特征")
            + ("☃" if index < 2 else "常规文本")
        )
        write_jsonl_atomic(
            posts_dir / f"{subject_id}.jsonl",
            [
                {"post_id": f"post-{index}-0", "cleaned_text": text + "OCD"},
                {"post_id": f"post-{index}-1", "cleaned_text": text},
            ],
        )
    write_jsonl_atomic(dataset_dir / "subjects.jsonl", subject_rows)
    return dataset_dir


def _write_split(root: Path) -> Path:
    subjects = [f"subject-{index:02d}" for index in range(10)]
    assignments: list[dict[str, object]] = []
    for fold in range(5):
        test_ids = {subjects[2 * fold], subjects[2 * fold + 1]}
        remaining = [subject for subject in subjects if subject not in test_ids]
        validation_ids = {
            next(subject for subject in remaining if int(subject[-2:]) % 2 == 0),
            next(subject for subject in remaining if int(subject[-2:]) % 2 == 1),
        }
        for subject_id in subjects:
            label = int(subject_id[-2:]) % 2
            assignments.append(
                {
                    "outer_fold": fold,
                    "subject_id": subject_id,
                    "label_name": "self_reported_ocd" if label else "control",
                    "label_id": label,
                    "role": (
                        "test"
                        if subject_id in test_ids
                        else "validation"
                        if subject_id in validation_ids
                        else "train"
                    ),
                    "available_post_count": 2,
                    "selected_post_count": 2,
                }
            )
    split_path = root / "splits" / "assignments.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_id": "splits-text-test",
                "dataset_id": "dataset-text-test",
                "seed": 99,
                "assignments": assignments,
            }
        ),
        encoding="utf-8",
    )
    return split_path


class TfidfLinearSVMTests(unittest.TestCase):
    def test_five_fold_run_uses_train_only_vocabulary_and_writes_oof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = train_tfidf_linear_svm(
                dataset=PreparedDataset(_write_synthetic_dataset(root)),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                output_root=root / "runs",
                configuration=TfidfLinearSVMConfig(minimum_document_frequency=1),
            )

            self.assertTrue(result.summary["complete_outer_cv"])
            self.assertEqual(result.summary["oof_metrics"]["n"], 10)
            self.assertEqual(len(result.summary["folds"]), 5)
            self.assertTrue((result.run_dir / "oof_predictions.jsonl").is_file())
            with (result.run_dir / "fold-0" / "vectorizer.json").open(encoding="utf-8") as handle:
                vocabulary = json.load(handle)["tokens_by_index"]
            self.assertNotIn("☃", vocabulary)
            import numpy as np

            model_path = result.run_dir / "fold-0" / "model_parameters.npz"
            with np.load(model_path, allow_pickle=False) as model:
                self.assertEqual(int(model["schema_version"][0]), 1)
                self.assertEqual(model["coefficient"].shape[0], 1)

            removed = train_tfidf_linear_svm(
                dataset=PreparedDataset(root / "dataset"),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                output_root=root / "runs",
                keyword_condition="removed",
                configuration=TfidfLinearSVMConfig(minimum_document_frequency=1),
            )
            self.assertEqual(removed.summary["keyword_condition"], "removed")
            removed_audit = json.loads(
                (removed.run_dir / "text_input_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(removed_audit["aggregate"]["source_selected_posts"], 20)
            self.assertEqual(removed_audit["aggregate"]["condition_posts"], 10)
            self.assertEqual(removed_audit["aggregate"]["keyword_matched_posts"], 10)
            self.assertEqual(removed_audit["aggregate"]["keyword_match_occurrences"], 10)
            with (removed.run_dir / "fold-0" / "vectorizer.json").open(encoding="utf-8") as handle:
                removed_vocabulary = json.load(handle)["tokens_by_index"]
            self.assertNotIn("o", removed_vocabulary)
            self.assertNotIn("oc", removed_vocabulary)

    def test_repeated_cv_summarizes_distinct_subject_level_oof_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = run_repeated_tfidf_linear_svm_cv(
                dataset=PreparedDataset(_write_synthetic_dataset(root)),
                study_config=load_config(study_path),
                run_output_root=root / "runs",
                split_output_root=root / "splits",
                repeated_cv_output_root=root / "repeated_cv",
                split_seeds=(42, 43),
                keyword_condition="removed",
                configuration=TfidfLinearSVMConfig(minimum_document_frequency=1),
            )
            self.assertEqual(result.summary["keyword_condition"], "removed")
            self.assertEqual(result.summary["n_split_repetitions"], 2)
            self.assertEqual(result.summary["outer_folds_per_repetition"], 5)
            self.assertEqual(result.summary["trained_fold_models"], 10)
            self.assertEqual(result.summary["oof_subjects_per_repetition"], 10)
            fingerprints = {
                row["outer_partition_sha256"] for row in result.summary["per_repetition"]
            }
            self.assertEqual(len(fingerprints), 2)
            self.assertTrue((result.repeated_cv_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
