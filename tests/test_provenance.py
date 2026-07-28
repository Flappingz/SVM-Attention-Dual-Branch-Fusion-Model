from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ocd_v3.provenance import create_run_manifest, git_state


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )


class ProvenanceTests(unittest.TestCase):
    def test_git_state_hashes_tracked_and_untracked_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            _git(repository, "init")
            _git(repository, "config", "user.name", "Synthetic Test")
            _git(repository, "config", "user.email", "synthetic@example.invalid")
            tracked = repository / "tracked.txt"
            tracked.write_text("version one\n", encoding="utf-8")
            _git(repository, "add", "tracked.txt")
            _git(repository, "commit", "-m", "initial")

            clean = git_state(repository)
            self.assertFalse(clean["dirty"])
            self.assertIsNone(clean["worktree_sha256"])

            tracked.write_text("version two\n", encoding="utf-8")
            tracked_change = git_state(repository)
            self.assertTrue(tracked_change["dirty"])
            self.assertIsNotNone(tracked_change["worktree_sha256"])

            untracked = repository / "new.py"
            untracked.write_text("value = 1\n", encoding="utf-8")
            first_manifest = create_run_manifest(
                repository=repository,
                dataset_id="dataset-test",
                split_id="splits-test",
                experiment_id="experiment-test",
                parameters={"value": 1},
                seeds={"training": 7},
            )
            untracked.write_text("value = 2\n", encoding="utf-8")
            second_manifest = create_run_manifest(
                repository=repository,
                dataset_id="dataset-test",
                split_id="splits-test",
                experiment_id="experiment-test",
                parameters={"value": 1},
                seeds={"training": 7},
            )
            self.assertNotEqual(first_manifest["run_id"], second_manifest["run_id"])
            self.assertNotEqual(
                first_manifest["source_code"]["worktree_sha256"],
                second_manifest["source_code"]["worktree_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
