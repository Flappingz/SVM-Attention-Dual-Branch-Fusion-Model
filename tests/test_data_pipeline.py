from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import POST_COLUMNS, make_raw_group, write_csv, write_study_config

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.data.prepare import prepare_dataset
from ocd_v3.data.raw import audit_raw_dataset


class DataPipelineTests(unittest.TestCase):
    def test_audit_and_prepare_use_current_raw_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100", "101"])
            make_raw_group(
                raw_root,
                "self_reporting_ocd",
                ["200", "201"],
                directory_users=["200"],
            )
            write_csv(
                raw_root / "self_reporting_ocd" / "200" / "200.csv",
                POST_COLUMNS,
                [
                    {
                        "id": "900000",
                        "正文": "我提到OCD但这不是临床标签",
                        "完整日期": "2026-01-02 03:04:05",
                        "点赞数": "4",
                        "评论数": "2",
                        "转发数": "1",
                    },
                    {
                        "id": "900000",
                        "正文": "我提到OCD但这不是临床标签",
                        "完整日期": "2026-01-02 03:04:05",
                        "点赞数": "9",
                        "评论数": "2",
                        "转发数": "1",
                    },
                ],
            )
            config_path = root / "study.json"
            write_study_config(config_path, raw_root, artifact_root)
            config = load_config(config_path)

            audit = audit_raw_dataset(config)
            positive = next(
                group for group in audit.groups if group.label_name == "self_reported_ocd"
            )
            self.assertEqual(positive.unique_listed_users, 2)
            self.assertEqual(positive.user_directories, 1)
            self.assertEqual(positive.listed_without_directory, 1)
            self.assertEqual(positive.duplicate_post_ids, 1)
            self.assertEqual(positive.subjects_with_duplicate_post_ids, 1)
            self.assertEqual(positive.conflicting_duplicate_post_ids, 1)

            dataset_dir, manifest = prepare_dataset(config)
            subjects_text = (dataset_dir / "subjects.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"uid"', subjects_text)
            self.assertNotIn('"100"', subjects_text)
            self.assertIn('"subject_id": "subject-', subjects_text)
            self.assertEqual(manifest["pseudonymization"]["method"], "HMAC-SHA256")
            self.assertEqual(manifest["groups"]["control"]["subjects_eligible"], 2)
            self.assertEqual(
                manifest["groups"]["self_reported_ocd"]["subjects_eligible"], 1
            )
            self.assertEqual(
                manifest["groups"]["self_reported_ocd"]["excluded_missing_user_directory"],
                1,
            )
            self.assertEqual(
                manifest["groups"]["self_reported_ocd"]["posts_duplicate_rows_dropped"],
                1,
            )

            dataset = PreparedDataset(dataset_dir)
            positive_subject = next(
                subject
                for subject in dataset.subjects()
                if subject.label_name == "self_reported_ocd"
            )
            original = dataset.posts(positive_subject.subject_id)
            masked = dataset.posts(
                positive_subject.subject_id,
                keyword_condition="masked",
                keywords=("ocd", "强迫症"),
            )
            self.assertIn("OCD", original[0]["cleaned_text"])
            self.assertNotIn("OCD", masked[0]["cleaned_text"])
            self.assertEqual(original[0]["post_id"], masked[0]["post_id"])
            self.assertNotEqual(original[0]["post_id"], "900000")
            self.assertNotIn("sid", original[0])
            self.assertEqual(original[0]["likes"], 9)


if __name__ == "__main__":
    unittest.main()
