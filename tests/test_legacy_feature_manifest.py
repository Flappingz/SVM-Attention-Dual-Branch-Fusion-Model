from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocd_v3.features.training_data import PreparedFeatureSet


class LegacyFeatureManifestTests(unittest.TestCase):
    def test_rejects_manifest_with_local_model_path_before_loading_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature_dir = Path(temporary)
            manifest = {
                "schema_version": 5,
                "feature_id": "features-legacy",
                "dataset_id": "dataset-test",
                "complete_dataset": True,
                "post_selection": {"maximum_posts_per_subject": 64},
                "encoder": {
                    "model_id": "Qwen/Qwen3-VL-Embedding-2B",
                    "local_snapshot": {
                        "config_path": "/private/model/config.json",
                        "weights_bytes": 123,
                    },
                },
            }
            (feature_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "legacy local model path"):
                PreparedFeatureSet(feature_dir)


if __name__ == "__main__":
    unittest.main()
