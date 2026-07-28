"""Check repository hygiene and prevent accidental tracking of research artifacts."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = frozenset(
    {
        "artifacts",
        "checkpoints",
        "data",
        "datasets",
        "features",
        "identity-mappings",
        "predictions",
        "private",
        "private-mappings",
        "raw",
        "runs",
        "splits",
    }
)
PRIVATE_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".csv",
        ".feather",
        ".h5",
        ".hdf5",
        ".jsonl",
        ".npy",
        ".npz",
        ".parquet",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".tsv",
    }
)
ABSOLUTE_PATH = re.compile(
    r"(?:\b[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+\\|"
    r"(?<![A-Za-z0-9_.-])/(?:home|Users|root|workspace|data|mnt/[A-Za-z])"
    r"(?:/|\b))"
)
TOKEN_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_PATH_EXCLUSIONS = frozenset(
    {
        Path(".gitignore"),
        Path("tools/repository_check.py"),
        Path("tools/release_audit.py"),
        Path("tests/test_repository_check_patterns.py"),
    }
)


def _skip_absolute_path_scan(relative: Path) -> bool:
    """Skip files whose purpose is to define or test repository scanning rules."""
    if relative in ABSOLUTE_PATH_EXCLUSIONS:
        return True
    return (
        len(relative.parts) >= 3
        and relative.parts[:2] == (".github", "workflows")
        and (
            relative.name.startswith("apply-")
            or relative.name.startswith("run-known-fixes")
        )
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def _git_blob(object_id: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def _tracked_paths() -> list[Path]:
    return [Path(item) for item in _git("ls-files", "-z").split("\0") if item]


def _check_paths(paths: list[Path], *, prefix: str = "") -> list[str]:
    issues: list[str] = []
    for relative in paths:
        if relative.parts and relative.parts[0] in PRIVATE_ROOTS:
            issues.append(
                f"{prefix}governed top-level path is tracked: {relative.as_posix()}"
            )
        if relative.suffix.lower() in PRIVATE_SUFFIXES:
            issues.append(
                f"{prefix}research artifact suffix is tracked: {relative.as_posix()}"
            )
    return issues


def _check_text(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for relative in paths:
        path = REPOSITORY / relative
        if not path.is_file():
            issues.append(f"tracked path is missing from worktree: {relative.as_posix()}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"non-text tracked file requires manual review: {relative.as_posix()}")
            continue
        if not _skip_absolute_path_scan(relative) and ABSOLUTE_PATH.search(content):
            issues.append(f"absolute local path found: {relative.as_posix()}")
        if any(pattern.search(content) for pattern in TOKEN_PATTERNS):
            issues.append(f"credential-like token found: {relative.as_posix()}")
    return issues


def _history_blob_entries() -> list[tuple[str, Path]]:
    candidates: set[tuple[str, Path]] = set()
    for line in _git("rev-list", "--objects", "--all").splitlines():
        object_id, separator, path_text = line.partition(" ")
        if separator and path_text:
            candidates.add((object_id, Path(path_text)))
    if not candidates:
        return []

    object_ids = sorted({object_id for object_id, _ in candidates})
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=REPOSITORY,
        check=True,
        text=True,
        input="\n".join(object_ids) + "\n",
        capture_output=True,
    )
    object_types = {
        object_id: object_type
        for object_id, object_type in (
            line.split(" ", maxsplit=1) for line in result.stdout.splitlines()
        )
    }
    return sorted(
        (object_id, path)
        for object_id, path in candidates
        if object_types.get(object_id) == "blob"
    )


def _check_history() -> tuple[list[str], int]:
    entries = _history_blob_entries()
    issues = _check_paths(
        sorted({path for _, path in entries}), prefix="history: "
    )
    scanned_objects: set[str] = set()
    for object_id, relative in entries:
        if object_id in scanned_objects:
            continue
        scanned_objects.add(object_id)
        payload = _git_blob(object_id)
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(
                "history: non-text blob requires manual review: "
                f"{relative.as_posix()} ({object_id[:12]})"
            )
            continue
        if not _skip_absolute_path_scan(relative) and ABSOLUTE_PATH.search(content):
            issues.append(
                "history: absolute local path found: "
                f"{relative.as_posix()} ({object_id[:12]})"
            )
        if any(pattern.search(content) for pattern in TOKEN_PATTERNS):
            issues.append(
                "history: credential-like token found: "
                f"{relative.as_posix()} ({object_id[:12]})"
            )
    return issues, len(scanned_objects)


def _check_ignore_rules() -> list[str]:
    issues: list[str] = []
    ignore_text = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")
    for directory in ("data", "datasets", "features", "runs", "private-mappings"):
        if f"/{directory}/" not in ignore_text:
            issues.append(f"missing root-anchored ignore rule: /{directory}/")
    probe = "src/ocd_v3/data/__repository_check_probe__"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", probe], cwd=REPOSITORY, check=False
    )
    if result.returncode == 0:
        issues.append(".gitignore accidentally ignores src/ocd_v3/data/")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan all reachable historical blobs and paths.",
    )
    args = parser.parse_args()

    paths = _tracked_paths()
    issues = _check_paths(paths) + _check_text(paths) + _check_ignore_rules()
    historical_blobs = 0
    if args.history:
        history_issues, historical_blobs = _check_history()
        issues.extend(history_issues)
    if issues:
        print("REPOSITORY CHECK FAILED", file=sys.stderr)
        for issue in sorted(set(issues)):
            print(f"- {issue}", file=sys.stderr)
        return 1
    suffix = (
        f"; {historical_blobs} unique historical blobs checked"
        if args.history
        else ""
    )
    print(f"REPOSITORY CHECK PASSED: {len(paths)} tracked files checked{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
