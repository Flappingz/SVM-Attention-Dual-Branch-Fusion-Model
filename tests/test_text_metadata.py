from __future__ import annotations

import math
import unittest
from datetime import datetime

from ocd_v3.features.metadata import TrainOnlyStandardizer, metadata_vector
from ocd_v3.features.text import clean_weibo_text, count_keyword_matches, mask_keywords


class TextAndMetadataTests(unittest.TestCase):
    def test_text_cleaning_and_masking(self) -> None:
        self.assertEqual(clean_weibo_text("分享图片"), "")
        self.assertEqual(clean_weibo_text("正文\n测试位置", "测试位置"), "正文")
        masked = mask_keywords("OCD和强迫症不是临床标签", ("ocd", "强迫症"))
        self.assertEqual(masked, "[MASK]和[MASK]不是临床标签")

    def test_keyword_match_count_agrees_with_masking_rule(self) -> None:
        text = "OCD、ocd、强迫症和強迫症"
        keywords = ("ocd", "强迫症", "強迫症")
        self.assertEqual(count_keyword_matches(text, keywords), 4)
        self.assertEqual(mask_keywords(text, keywords).count("[MASK]"), 4)

    def test_metadata_is_log_scaled_and_cyclical(self) -> None:
        vector = metadata_vector(3, 0, 1, datetime(2026, 1, 1, 6, 0, 0))
        self.assertAlmostEqual(vector[0], math.log(4))
        self.assertAlmostEqual(vector[3], 1.0)
        self.assertAlmostEqual(vector[4], 0.0, places=7)

    def test_standardizer_is_explicitly_fit(self) -> None:
        scaler = TrainOnlyStandardizer.fit([(1, 2), (3, 4)])
        self.assertEqual(scaler.transform((2, 3)), (0.0, 0.0))

    def test_empty_standardizer_represents_explicit_metadata_exclusion(self) -> None:
        scaler = TrainOnlyStandardizer.fit([(), (), ()])
        self.assertEqual(scaler.mean, ())
        self.assertEqual(scaler.scale, ())
        self.assertEqual(scaler.transform(()), ())


if __name__ == "__main__":
    unittest.main()
