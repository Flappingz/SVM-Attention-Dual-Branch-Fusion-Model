from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ocd_v3.data.dataset import SubjectRecord
from ocd_v3.evaluation.splits import (
    assign_outer_folds,
    build_nested_assignments,
    create_split_artifact,
    validate_assignments,
)


class _SyntheticDataset:
    dataset_id = "dataset-test"

    def __init__(self, subjects: list[SubjectRecord]) -> None:
        self._subjects = subjects

    def subjects(self) -> list[SubjectRecord]:
        return self._subjects


class SplitTests(unittest.TestCase):
    def test_each_subject_is_tested_once_without_leakage(self) -> None:
        subjects = [
            SubjectRecord(
                subject_id=f"sub-{label}-{index}",
                label_name="control" if label == 0 else "self_reported_ocd",
                label_id=label,
                post_count=10 + index,
                media_count=index,
            )
            for label in (0, 1)
            for index in range(10)
        ]
        outer = assign_outer_folds(subjects, fold_count=5, seed=17)
        assignments = build_nested_assignments(
            subjects,
            outer_folds=outer,
            fold_count=5,
            validation_fraction=0.125,
            seed=17,
        )
        validate_assignments(assignments, subjects, fold_count=5)
        test_counts = Counter(
            row.subject_id for row in assignments if row.role == "test"
        )
        self.assertEqual(set(test_counts.values()), {1})
        for fold in range(5):
            test_rows = [
                row for row in assignments if row.outer_fold == fold and row.role == "test"
            ]
            self.assertEqual(Counter(row.label_id for row in test_rows), {0: 2, 1: 2})

    def test_split_seed_changes_partition(self) -> None:
        subjects = [
            SubjectRecord(
                subject_id=f"sub-{label}-{index}",
                label_name="control" if label == 0 else "self_reported_ocd",
                label_id=label,
                post_count=64,
                media_count=index,
            )
            for label in (0, 1)
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "splits"
            first_path, first = create_split_artifact(
                dataset=_SyntheticDataset(subjects),
                fold_count=5,
                validation_fraction=0.125,
                seed=42,
                output_root=output_root,
            )
            second_path, second = create_split_artifact(
                dataset=_SyntheticDataset(subjects),
                fold_count=5,
                validation_fraction=0.125,
                seed=43,
                output_root=output_root,
            )
            self.assertNotEqual(first["split_id"], second["split_id"])
            first_rows = json.loads(first_path.read_text(encoding="utf-8"))["assignments"]
            second_rows = json.loads(second_path.read_text(encoding="utf-8"))["assignments"]
            first_test = {
                row["subject_id"]: row["outer_fold"]
                for row in first_rows
                if row["role"] == "test"
            }
            second_test = {
                row["subject_id"]: row["outer_fold"]
                for row in second_rows
                if row["role"] == "test"
            }
            self.assertNotEqual(first_test, second_test)

    def test_minimum_posts_filter_is_applied_before_splitting(self) -> None:
        subjects = [
            SubjectRecord(
                subject_id=f"sub-{label}-{index}",
                label_name="control" if label == 0 else "self_reported_ocd",
                label_id=label,
                post_count=10 if index == 0 else 20 + index,
                media_count=0,
            )
            for label in (0, 1)
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            split_path, summary = create_split_artifact(
                dataset=_SyntheticDataset(subjects),
                fold_count=5,
                validation_fraction=0.125,
                seed=42,
                output_root=Path(temporary),
                minimum_posts_per_subject=20,
            )
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            included_ids = {row["subject_id"] for row in payload["assignments"]}
            self.assertNotIn("sub-0-0", included_ids)
            self.assertNotIn("sub-1-0", included_ids)
            self.assertEqual(len(included_ids), 10)
            self.assertEqual(payload["minimum_available_posts_per_subject"], 20)
            self.assertEqual(summary["cohort"]["subjects_before_post_filter"], 12)
            self.assertEqual(summary["cohort"]["subjects_included"], 10)
            self.assertEqual(summary["cohort"]["subjects_excluded_by_post_filter"], 2)
            self.assertEqual(summary["cohort"]["excluded_labels"], {
                "control": 1,
                "self_reported_ocd": 1,
            })

    def test_minimum_posts_filter_rejects_nonpositive_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must be positive"):
                create_split_artifact(
                    dataset=_SyntheticDataset([]),
                    fold_count=5,
                    validation_fraction=0.125,
                    seed=42,
                    output_root=Path(temporary),
                    minimum_posts_per_subject=0,
                )


if __name__ == "__main__":
    unittest.main()
