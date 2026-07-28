from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import write_study_config

from ocd_v3.config import ConfigurationError, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_strict_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "study.json"
            write_study_config(config_path, root / "raw", root / "artifacts")
            config = load_config(config_path)
            self.assertEqual(config.evaluation.outer_folds, 5)
            self.assertEqual(config.dataset.post_selection.maximum_posts_per_subject, 64)
            self.assertEqual(
                config.resolved_dict()["raw_root"], str((root / "raw").resolve())
            )
            self.assertEqual(
                config.resolved_dict()["artifact_root"],
                str((root / "artifacts").resolve()),
            )
            self.assertNotIn("raw_root", config.public_dict())
            self.assertNotIn("artifact_root", config.public_dict())
            self.assertNotIn("private_mapping_root", config.public_dict())

            second_path = root / "study-other-paths.json"
            write_study_config(
                second_path,
                root / "raw-other",
                root / "artifacts-other",
            )
            second_config = load_config(second_path)
            self.assertEqual(config.digest(), second_config.digest())

    def test_missing_environment_variable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            path.write_text(
                (
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "study.keyword-post-removed.example.json"
                ).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ConfigurationError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
