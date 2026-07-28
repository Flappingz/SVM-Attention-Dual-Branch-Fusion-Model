from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocd_v3.provenance import directory_content_fingerprint


class RuntimeIdentityTests(unittest.TestCase):
    def test_attention_backend_changes_model_snapshot_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "model.safetensors").write_bytes(b"synthetic-weights")

            with patch.dict(
                os.environ,
                {"QWEN3_VL_ATTN_IMPLEMENTATION": "eager"},
                clear=False,
            ):
                eager = directory_content_fingerprint(model_path)
            with patch.dict(
                os.environ,
                {"QWEN3_VL_ATTN_IMPLEMENTATION": "sdpa"},
                clear=False,
            ):
                sdpa = directory_content_fingerprint(model_path)

            self.assertEqual(eager["sha256"], sdpa["sha256"])
            self.assertEqual(
                eager["inference_environment"]["qwen3_vl_attention_implementation"],
                "eager",
            )
            self.assertEqual(
                sdpa["inference_environment"]["qwen3_vl_attention_implementation"],
                "sdpa",
            )
            self.assertNotEqual(eager, sdpa)

    def test_same_size_model_change_updates_content_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            weights = model_path / "model.safetensors"
            weights.write_bytes(b"abc")
            first = directory_content_fingerprint(model_path)
            weights.write_bytes(b"abd")
            second = directory_content_fingerprint(model_path)

            self.assertEqual(first["total_bytes"], second["total_bytes"])
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
