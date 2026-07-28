from __future__ import annotations

import unittest

from ocd_v3.evaluation.metrics import binary_metrics, mean_sd_ci95
from ocd_v3.evaluation.oof import OOFPrediction, evaluate_oof


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_and_auc(self) -> None:
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)
        self.assertAlmostEqual(metrics.roc_auc or 0.0, 0.75)
        self.assertEqual((metrics.tp, metrics.tn, metrics.fp, metrics.fn), (1, 2, 0, 1))

    def test_fold_confidence_interval_uses_sample_sd(self) -> None:
        summary = mean_sd_ci95([1, 2, 3, 4, 5])
        self.assertEqual(summary.n, 5)
        self.assertEqual(summary.mean, 3)
        self.assertGreater(summary.ci95_high, summary.mean)

    def test_oof_requires_exactly_one_prediction_per_subject(self) -> None:
        rows = [
            OOFPrediction("sub-a", 0, 0, 0.1),
            OOFPrediction("sub-b", 1, 1, 0.9),
        ]
        metrics = evaluate_oof(rows, {"sub-a", "sub-b"}, threshold=0.5)
        self.assertEqual(metrics.accuracy, 1.0)
        with self.assertRaises(ValueError):
            evaluate_oof(rows + [rows[0]], {"sub-a", "sub-b"}, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
