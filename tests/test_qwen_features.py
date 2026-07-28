from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocd_v3.features.qwen_vl import (
    LocalMediaDecodeError,
    configured_attention_implementation,
    evenly_sample_paths,
    load_image_copy,
)


class QwenFeatureTests(unittest.TestCase):
    def test_attention_implementation_is_opt_in_and_validated(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(configured_attention_implementation())
        with patch.dict(
            "os.environ", {"QWEN3_VL_ATTN_IMPLEMENTATION": "flash_attention_2"}, clear=True
        ):
            self.assertEqual(configured_attention_implementation(), "flash_attention_2")
        with patch.dict("os.environ", {"QWEN3_VL_ATTN_IMPLEMENTATION": "invalid"}, clear=True):
            with self.assertRaises(ValueError):
                configured_attention_implementation()

    def test_media_sampling_is_deterministic_and_order_preserving(self) -> None:
        paths = [Path(f"frame-{index}.jpg") for index in range(7)]
        selected = evenly_sample_paths(paths, 3)
        self.assertEqual(selected, (paths[0], paths[3], paths[6]))
        self.assertEqual(evenly_sample_paths(paths, 0), ())

    def test_media_decode_error_includes_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private-user-id.jpg"
            path.write_bytes(b"invalid-image")
            with self.assertRaises(LocalMediaDecodeError) as context:
                load_image_copy(path)
            self.assertIn(str(path), str(context.exception))


if __name__ == "__main__":
    unittest.main()
