from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.features.text import contains_keywords


class DatasetKeywordConditionTests(unittest.TestCase):
    def test_keyword_post_removal_example_includes_requested_traditional_terms(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "study.keyword-post-removed.example.json"
        )
        keywords = json.loads(config_path.read_text(encoding="utf-8"))["dataset"]["keywords"]
        self.assertIn("\u5f37\u8feb\u969c\u7919", keywords)
        self.assertIn("\u5f37\u8feb\u884c\u70ba", keywords)
        self.assertIn("\u5f37\u8feb\u601d\u7dad", keywords)

    def test_removed_condition_filters_within_recent_post_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "posts").mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "dataset-test",
                        "post_selection_for_models": {
                            "maximum_posts_per_subject": 2,
                            "strategy": "most_recent",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "subjects.jsonl").write_text("", encoding="utf-8")
            posts = [
                {"post_id": "post-1", "cleaned_text": "first ordinary post"},
                {"post_id": "post-2", "cleaned_text": "OCD mention"},
                {"post_id": "post-3", "cleaned_text": "last ordinary post"},
            ]
            (root / "posts" / "subject.jsonl").write_text(
                "\n".join(json.dumps(row) for row in posts) + "\n",
                encoding="utf-8",
            )

            dataset = PreparedDataset(root)
            selected = dataset.posts(
                "subject",
                maximum_posts=2,
                keyword_condition="removed",
                keywords=("ocd",),
            )

            self.assertEqual([row["post_id"] for row in selected], ["post-3"])

    def test_keyword_detection_is_case_insensitive_for_english_and_handles_chinese(self) -> None:
        keywords = ("ocd", "\u5f3a\u8feb\u75c7", "\u5f37\u8feb\u75c7")
        self.assertTrue(contains_keywords("mentioning oCd", keywords))
        self.assertTrue(contains_keywords("\u5f3a\u8feb\u75c7", keywords))
        self.assertTrue(contains_keywords("\u5f37\u8feb\u75c7", keywords))
        self.assertFalse(contains_keywords("ordinary post", keywords))


if __name__ == "__main__":
    unittest.main()
