from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from helpers import write_study_config
from test_tfidf_linear_svm import _write_split

from ocd_v3.config import load_config
from ocd_v3.experiments.metadata_logistic import train_metadata_logistic
from ocd_v3.features.storage import save_subject_feature_bundle
from ocd_v3.features.training_data import PreparedFeatureSet


def _write_feature_set(root: Path) -> Path:
    feature_dir = root / "features"
    subject_dir = feature_dir / "subjects"
    for index in range(10):
        subject_id = f"subject-{index:02d}"
        label = index % 2
        # Fold-0 test subjects have extreme values and must not affect its scaler.
        offset = 100.0 if index < 2 else float(label)
        save_subject_feature_bundle(
            subject_dir / f"{subject_id}.npz",
            subject_id=subject_id,
            post_ids=[f"post-{index}-0", f"post-{index}-1"],
            metadata=[
                [offset, label, 0.0, 1.0, 0.0],
                [offset + 1.0, label, 1.0, 0.0, 1.0],
            ],
            content_embeddings=[[[0.0] * 8], [[0.0] * 8]],
            representation_names=["final"],
            modality_presence=[[1, 0, 0], [1, 0, 0]],
            modality_names=["text", "image", "live_photo"],
            encoder_manifest={"model_id": "synthetic", "revision": "test"},
        )
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "feature_id": "features-metadata-test",
                "dataset_id": "dataset-text-test",
                "keyword_condition": "removed",
                "complete_dataset": True,
                "post_selection": {
                    "maximum_posts_per_subject": 2,
                    "strategy": "most_recent",
                },
                "metadata_features": [
                    "log1p_likes",
                    "log1p_reposts",
                    "log1p_comments",
                    "time_sin",
                    "time_cos",
                ],
            }
        ),
        encoding="utf-8",
    )
    return feature_dir


@unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is absent")
class MetadataLogisticTests(unittest.TestCase):
    def test_uses_train_post_scaler_and_writes_complete_oof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = train_metadata_logistic(
                feature_set=PreparedFeatureSet(_write_feature_set(root)),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                output_root=root / "runs",
            )

            self.assertTrue(result.summary["complete_outer_cv"])
            self.assertEqual(result.summary["oof_metrics"]["n"], 10)
            self.assertEqual(result.summary["metadata_condition"], "only")
            self.assertEqual(
                result.summary["folds"][0]["model_parameter_count"], 6
            )
            scaler = json.loads(
                (result.run_dir / "fold-0" / "metadata_standardizer.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertLess(float(scaler["mean"][0]), 10.0)
            self.assertTrue((result.run_dir / "oof_predictions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
