from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocd_v3.io import write_json_atomic


def _git_text(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_bytes(repository: Path, *arguments: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _update_file_hash(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def _worktree_sha256(repository: Path, dirty: bool) -> str | None:
    if not dirty:
        return None
    tracked_diff = _git_bytes(repository, "diff", "--binary", "HEAD", "--")
    untracked = _git_bytes(
        repository, "ls-files", "--others", "--exclude-standard", "-z"
    )
    if tracked_diff is None or untracked is None:
        return None
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    for raw_relative in sorted(value for value in untracked.split(b"\0") if value):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        path = repository / relative
        digest.update(b"\0untracked-path\0")
        digest.update(raw_relative)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"file\0")
            _update_file_hash(digest, path)
        else:
            digest.update(b"other\0")
    return digest.hexdigest()


def directory_content_fingerprint(root: Path) -> dict[str, Any]:
    """Hash a local model snapshot and record content-affecting inference settings."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Configured model directory does not exist")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError("Configured model directory contains no files")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        _update_file_hash(digest, path)
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "inference_environment": {
            "qwen3_vl_attention_implementation": os.environ.get(
                "QWEN3_VL_ATTN_IMPLEMENTATION"
            ),
        },
    }


def git_state(repository: Path) -> dict[str, Any]:
    """Record the commit and a content hash for any local source changes."""
    repository = repository.resolve()
    status = _git_text(
        repository, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    dirty = bool(status) if status is not None else None
    return {
        "commit": _git_text(repository, "rev-parse", "HEAD"),
        "branch": _git_text(repository, "branch", "--show-current"),
        "dirty": dirty,
        "worktree_sha256": (
            _worktree_sha256(repository, dirty) if dirty is not None else None
        ),
    }


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> dict[str, Any]:
    details: dict[str, Any] = {
        "processor": platform.processor() or None,
        "machine": platform.machine(),
    }
    try:
        import torch

        cuda: dict[str, Any] = {
            "available": torch.cuda.is_available(),
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
        details["cuda"] = cuda
    except ImportError:
        details["cuda"] = {"available": False, "reason": "torch_not_installed"}
    return details


def create_run_manifest(
    *,
    repository: Path,
    dataset_id: str,
    split_id: str,
    experiment_id: str,
    parameters: dict[str, Any],
    seeds: dict[str, int],
) -> dict[str, Any]:
    source_git = git_state(repository)
    identity = {
        "dataset_id": dataset_id,
        "split_id": split_id,
        "experiment_id": experiment_id,
        "parameters": parameters,
        "seeds": seeds,
        "source_code": {
            "commit": source_git["commit"],
            "dirty": source_git["dirty"],
            "worktree_sha256": source_git["worktree_sha256"],
        },
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": 3,
        "run_id": f"run-{run_id}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
        "git": source_git,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "packages": _package_versions(
                [
                    "numpy",
                    "scikit-learn",
                    "torch",
                    "transformers",
                    "Pillow",
                    "qwen-vl-utils",
                    "safetensors",
                    "av",
                    "ijson",
                ]
            ),
            "hardware": _hardware(),
        },
    }


def save_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json_atomic(path, manifest)
