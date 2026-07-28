from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ocd_v3.data.dataset import SubjectRecord
from ocd_v3.experiments.controlled_benchmark import (
    ControlledSplitRecord,
    _validate_protocol,
    audit_model_oof_alignment,
    create_controlled_splits,
    matched_model_comparisons,
)
from ocd_v3.experiments.controlled_benchmark_config import (
    CONTROLLED_MODEL_IDS,
    ControlledBenchmarkConfig,
    ControlledCohortConfig,
    ControlledSplitConfig,
    load_controlled_benchmark_config,
)
from ocd_v3.io import write_jsonl_atomic


def _protocol() -> ControlledBenchmarkConfig:
    return ControlledBenchmarkConfig(
        schema_version=1,
        expected_dataset_id="dataset-controlled-test",
        keyword_condition="removed",
        cohort=ControlledCohortConfig(
            field="available_post_count", operator=">", threshold=20
        ),
        splits=ControlledSplitConfig(
            seeds=(42, 43, 44, 45, 46, 47, 48, 49),
            outer_folds=5,
            validation_fraction_within_outer_train=0.125,
        ),
        training_base_seed=42,
        classification_threshold=0.5,
        primary_metric="f1",
        secondary_metrics=("roc_auc", "accuracy", "precision", "recall"),
        models=CONTROLLED_MODEL_IDS,
    )


class _SyntheticDataset:
    dataset_id = "dataset-controlled-test"

    def __init__(self) -> None:
        self._subjects = [
            SubjectRecord(
                subject_id=f"subject-{label}-{index}",
                label_name="self_reported_ocd" if label else "control",
                label_id=label,
                post_count=20 if index == 0 else 20 + index,
                media_count=0,
            )
            for label in (0, 1)
            for index in range(6)
        ]

    def subjects(self) -> list[SubjectRecord]:
        return self._subjects


class ControlledBenchmarkTests(unittest.TestCase):
    def test_frozen_config_loads_strict_gt20_and_eight_by_five_protocol(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "controlled_benchmark.keyword-post-removed.gt20.json"
        )
        configuration = load_controlled_benchmark_config(path)
        self.assertIsNone(configuration.expected_dataset_id)
        self.assertEqual(configuration.cohort.minimum_posts_per_subject, 21)
        self.assertEqual(configuration.splits.outer_folds, 5)
        self.assertEqual(len(configuration.splits.seeds), 8)
        self.assertEqual(configuration.models, CONTROLLED_MODEL_IDS)

    def test_null_dataset_id_skips_only_the_exact_id_pin(self) -> None:
        base = _protocol()
        configuration = ControlledBenchmarkConfig(
            **{**base.__dict__, "expected_dataset_id": None}
        )
        encoder = SimpleNamespace(
            model_id="qwen", revision="revision", representations=("final",), normalize=True
        )
        qwen_features = SimpleNamespace(
            dataset_id="dataset-local",
            manifest={
                "keyword_condition": "removed",
                "complete_dataset": True,
                "encoder": {
                    "model_id": "qwen",
                    "revision": "revision",
                    "representations": ["final"],
                    "normalize": True,
                },
            },
        )
        chinese_features = SimpleNamespace(
            dataset_id="dataset-local",
            manifest={
                "keyword_condition": "removed",
                "complete_dataset": True,
                "encoder": {"model_id": "roberta", "revision": "revision"},
            },
        )
        _validate_protocol(
            benchmark_config=configuration,
            dataset=SimpleNamespace(dataset_id="dataset-local"),
            study_config=SimpleNamespace(
                evaluation=SimpleNamespace(
                    outer_folds=5,
                    validation_fraction_within_outer_train=0.125,
                    classification_threshold=0.5,
                )
            ),
            qwen_feature_set=qwen_features,
            chinese_feature_set=chinese_features,
            full_config=SimpleNamespace(
                encoder=encoder, training=SimpleNamespace(seed=42)
            ),
            qwen_config=SimpleNamespace(
                encoder=encoder, training=SimpleNamespace(seed=42)
            ),
            chinese_config=SimpleNamespace(
                encoder=SimpleNamespace(model_id="roberta", revision="revision"),
                classifier=SimpleNamespace(random_state=42),
            ),
            tfidf_config=SimpleNamespace(random_state=42),
        )

    def test_shared_splits_exclude_exactly_twenty_post_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records = create_controlled_splits(
                dataset=_SyntheticDataset(),
                study_config=SimpleNamespace(split_dir=Path(temporary)),
                benchmark_config=_protocol(),
            )
            self.assertEqual(len(records), 8)
            self.assertEqual(len({record.summary["split_id"] for record in records}), 8)
            for record in records:
                payload = json.loads(record.assignments_path.read_text(encoding="utf-8"))
                subject_ids = {row["subject_id"] for row in payload["assignments"]}
                self.assertNotIn("subject-0-0", subject_ids)
                self.assertNotIn("subject-1-0", subject_ids)
                self.assertEqual(len(subject_ids), 10)
                self.assertEqual(payload["minimum_available_posts_per_subject"], 21)

    def test_matched_comparisons_use_identical_seed_and_split_order(self) -> None:
        summaries: dict[str, dict[str, object]] = {}
        baselines = {
            "tfidf_linear_svm": 0.60,
            "chinese_roberta_mean_logistic": 0.58,
            "qwen_vl_mean_mlp": 0.62,
            "hierarchical_attention_full": 0.66,
        }
        for model_id, level in baselines.items():
            rows = []
            for offset, seed in enumerate(range(42, 50)):
                metrics = {
                    name: level + 0.001 * offset + 0.0001 * metric_index
                    for metric_index, name in enumerate(
                        ("f1", "roc_auc", "accuracy", "precision", "recall")
                    )
                }
                rows.append(
                    {
                        "split_seed": seed,
                        "split_id": f"split-{seed}",
                        "oof_metrics": metrics,
                    }
                )
            summaries[model_id] = {"per_repetition": rows}

        result = matched_model_comparisons(summaries)
        tfidf_f1 = result["by_baseline"]["tfidf_linear_svm"]["f1"]
        self.assertAlmostEqual(tfidf_f1["summary"]["mean"], 0.06)
        self.assertEqual(tfidf_f1["wins"], 8)
        self.assertEqual(tfidf_f1["losses"], 0)
        self.assertLessEqual(tfidf_f1["holm_adjusted_p_within_metric"], 1.0)

    def test_oof_audit_checks_all_four_models_against_all_shared_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_records: list[ControlledSplitRecord] = []
            expected_rows_by_seed: dict[int, list[dict[str, object]]] = {}
            for seed in range(42, 50):
                assignments = []
                predictions = []
                for index in range(10):
                    subject_id = f"subject-{index}"
                    label = index % 2
                    fold = (index + seed) % 5
                    assignments.append(
                        {
                            "subject_id": subject_id,
                            "label_id": label,
                            "outer_fold": fold,
                            "role": "test",
                        }
                    )
                    predictions.append(
                        {
                            "subject_id": subject_id,
                            "label": label,
                            "outer_fold": fold,
                            "score": 0.75 if label else 0.25,
                        }
                    )
                split_id = f"split-{seed}"
                split_path = root / "splits" / split_id / "assignments.json"
                split_path.parent.mkdir(parents=True)
                split_path.write_text(
                    json.dumps({"split_id": split_id, "assignments": assignments}),
                    encoding="utf-8",
                )
                split_records.append(
                    ControlledSplitRecord(
                        seed=seed,
                        assignments_path=split_path,
                        summary={"split_id": split_id},
                    )
                )
                expected_rows_by_seed[seed] = predictions

            runs_by_model: dict[str, list[SimpleNamespace]] = {}
            for model_id in CONTROLLED_MODEL_IDS:
                runs = []
                for seed in range(42, 50):
                    run_dir = root / "runs" / model_id / str(seed)
                    run_dir.mkdir(parents=True)
                    split_id = f"split-{seed}"
                    (run_dir / "summary.json").write_text(
                        json.dumps({"split_id": split_id}), encoding="utf-8"
                    )
                    (run_dir / "run_manifest.json").write_text(
                        json.dumps({"split_id": split_id}), encoding="utf-8"
                    )
                    write_jsonl_atomic(
                        run_dir / "oof_predictions.jsonl", expected_rows_by_seed[seed]
                    )
                    runs.append(SimpleNamespace(run_dir=run_dir))
                runs_by_model[model_id] = runs

            audit = audit_model_oof_alignment(
                split_records=split_records, runs_by_model=runs_by_model
            )
            self.assertTrue(audit["exact_split_subject_label_fold_alignment"])
            self.assertEqual(audit["outer_folds_per_repetition"], 5)
            self.assertEqual(audit["oof_prediction_rows_checked"], 4 * 8 * 10)


if __name__ == "__main__":
    unittest.main()
