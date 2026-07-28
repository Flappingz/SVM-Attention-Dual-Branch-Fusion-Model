"""Private identity-mapping support for governed local execution.

These mapping files are intentionally stored outside both the repository and the
regular artifact root.  They are needed only to resolve pseudonymous media records
back to locally governed source files during feature extraction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.io import read_jsonl, write_jsonl_atomic

PRIVATE_MAPPING_SCHEMA_VERSION = 1


def mapping_directory(config: StudyConfig, dataset_id: str) -> Path:
    return config.private_mapping_dir / dataset_id


def write_private_mapping(
    directory: Path, dataset_id: str, rows: Iterable[dict[str, Any]]
) -> None:
    """Write the raw-identity map in a dedicated, non-repository location."""
    directory.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(directory / "identity-map.jsonl", rows)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_MAPPING_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "contains_raw_identity_mapping": True,
                "do_not_version_control": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_private_media_paths(
    config: StudyConfig, dataset_id: str, subject_id: str
) -> dict[str, Path]:
    """Resolve pseudonymous media IDs through the external identity map."""
    directory = mapping_directory(config, dataset_id)
    manifest_path = directory / "manifest.json"
    mapping_path = directory / "identity-map.jsonl"
    if not manifest_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            "The private identity mapping is unavailable. Re-run prepare in the governed "
            "environment with the same pseudonymization key."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != PRIVATE_MAPPING_SCHEMA_VERSION
        or manifest.get("dataset_id") != dataset_id
        or not manifest.get("contains_raw_identity_mapping")
    ):
        raise ValueError("Private identity mapping does not match the prepared dataset")

    raw_root = config.raw_root.resolve()
    for row in read_jsonl(mapping_path):
        if str(row.get("subject_id")) != subject_id:
            continue
        result: dict[str, Path] = {}
        media_rows = row.get("media")
        if not isinstance(media_rows, list):
            raise ValueError("Private identity mapping has an invalid media section")
        for media in media_rows:
            if not isinstance(media, dict):
                raise ValueError("Private identity mapping has an invalid media record")
            media_id = str(media.get("media_id", ""))
            relative_path = media.get("source_relative_path")
            if not media_id or not isinstance(relative_path, str):
                raise ValueError("Private identity mapping has an incomplete media record")
            source_path = (raw_root / relative_path).resolve()
            if source_path != raw_root and raw_root not in source_path.parents:
                raise ValueError("Private media path escapes the configured raw root")
            if media_id in result:
                raise ValueError("Private identity mapping has duplicate media IDs")
            result[media_id] = source_path
        return result
    raise ValueError("Private identity mapping has no record for the requested subject")
