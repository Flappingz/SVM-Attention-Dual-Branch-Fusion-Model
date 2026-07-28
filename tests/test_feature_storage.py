from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from ocd_v3.features.storage import (
    load_subject_feature_bundle,
    save_subject_feature_bundle,
)


@unittest.skipIf(importlib.util.find_spec("numpy") is None, "NumPy is not installed")
class FeatureStorageTests(unittest.TestCase):
    def test_npz_round_trip_disallows_object_pickle(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subject.npz"
            save_subject_feature_bundle(
                path,
                subject_id="sub-test",
                post_ids=["post-a", "post-b"],
                metadata=[[0, 1], [2, 3]],
                content_embeddings=np.zeros((2, 1, 4), dtype=np.float32),
                representation_names=["final"],
                modality_presence=[[True, False, False], [True, True, False]],
                modality_names=["text", "image", "live_photo"],
                encoder_manifest={"encoder": "synthetic"},
            )
            payload = load_subject_feature_bundle(path)
            self.assertEqual(payload["content_embeddings"].shape, (2, 1, 4))
            self.assertEqual(payload["modality_presence"].shape, (2, 3))
            self.assertEqual(payload["encoder_manifest"]["encoder"], "synthetic")


if __name__ == "__main__":
    unittest.main()
