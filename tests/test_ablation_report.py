from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocd_v3.experiments.ablation_report import (
    ARCHITECTURE_CONDITIONS,
    METRICS,
    MODALITY_CONDITIONS,
    build_ablation_report,
)


def _summary(*, offset: float, run_prefix: str) -> dict[str, object]:
    seeds = list(range(58, 66))
    split_ids = [f"split-{seed}" for seed in seeds]
    rows = []
    values = {metric: [] for metric in METRICS}
    for index, (seed, split_id) in enumerate(zip(seeds, split_ids, strict=True)):
        metrics = {
            metric: 0.65 + index * 0.001 + offset for metric in METRICS
        }
        for metric, value in metrics.items():
            values[metric].append(value)
        rows.append(
            {
                "split_seed": seed,
                "split_id": split_id,
                "run_id": f"{run_prefix}-{seed}",
                "oof_metrics": metrics,
            }
        )
    return {
        "dataset_id": "dataset-test",
        "keyword_condition": "removed",
        "classification_threshold": 0.5,
        "split_seeds": seeds,
        "split_ids": split_ids,
        "run_ids": [row["run_id"] for row in rows],
        "oof_subjects_per_repetition": 177,
        "metric_summary": {
            metric: {
                "n": 8,
                "mean": sum(metric_values) / 8,
                "sample_sd": 0.001,
                "ci95_low": min(metric_values),
                "ci95_high": max(metric_values),
            }
            for metric, metric_values in values.items()
        },
        "per_repetition": rows,
    }


class AblationReportTests(unittest.TestCase):
    def test_requires_matched_protocol_and_builds_paired_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = _summary(offset=0.0, run_prefix="full")
            modality_paths: dict[str, Path] = {}
            for index, name in enumerate(MODALITY_CONDITIONS):
                payload = full if name == "text+image+metadata" else _summary(
                    offset=-0.01 * (index + 1), run_prefix=name
                )
                path = root / f"modality-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                modality_paths[name] = path
            architecture_paths: dict[str, Path] = {}
            for index, name in enumerate(ARCHITECTURE_CONDITIONS):
                payload = full if name == "full" else _summary(
                    offset=-0.005 * index, run_prefix=name
                )
                path = root / f"architecture-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                architecture_paths[name] = path

            output_path, report = build_ablation_report(
                modality_summary_paths=modality_paths,
                architecture_summary_paths=architecture_paths,
                output_dir=root / "report",
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue((root / "report" / "report.md").is_file())
            contrast = report["architecture"]["paired_comparisons"]["by_condition"][
                "without_post_cross_attention"
            ]["f1"]
            self.assertAlmostEqual(contrast["summary"]["mean"], -0.005)
            self.assertEqual(contrast["wins"], 0)
            self.assertEqual(contrast["losses"], 8)


if __name__ == "__main__":
    unittest.main()
