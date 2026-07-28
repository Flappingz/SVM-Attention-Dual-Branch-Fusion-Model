from __future__ import annotations

import runpy
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


class RepositoryCheckPatternTests(unittest.TestCase):
    def test_absolute_path_pattern_ignores_repository_relative_data_paths(self) -> None:
        namespace = runpy.run_path(str(REPOSITORY / "tools" / "repository_check.py"))
        pattern = namespace["ABSOLUTE_PATH"]
        separator = chr(92)
        windows_path = separator.join(("C:", "Users", "researcher", "file.txt"))

        self.assertIsNone(pattern.search("src/ocd_v3/data/file.py"))
        self.assertIsNone(pattern.search("../data/file.csv"))
        self.assertIsNotNone(pattern.search("/" + "data/private/file.csv"))
        self.assertIsNotNone(pattern.search(windows_path))

    def test_ci_propagates_failures_from_logged_commands(self) -> None:
        workflow = (REPOSITORY / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )

        self.assertGreaterEqual(workflow.count("set -o pipefail"), 3)
        self.assertIn("repository_check.py --history", workflow)
        self.assertIn("fetch-depth: 0", workflow)


if __name__ == "__main__":
    unittest.main()
