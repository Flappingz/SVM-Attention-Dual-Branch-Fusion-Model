from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocd_v3.config import GroupConfig, StudyConfig
from ocd_v3.data.private_mapping import mapping_directory, write_private_mapping
from ocd_v3.data.raw import (
    COMMENTS_COLUMN,
    LIKES_COLUMN,
    POST_DATE_COLUMN,
    POST_DATE_FALLBACK_COLUMN,
    POST_ID_COLUMN,
    POST_LOCATION_COLUMN,
    POST_TEXT_COLUMN,
    REPOSTS_COLUMN,
    USER_ID_COLUMN,
    RawAuditReport,
    RawDataError,
    audit_raw_dataset,
    iter_csv,
    iter_subject_media,
    read_csv,
    sha256_file,
)
from ocd_v3.features.metadata import (
    metadata_vector,
    parse_nonnegative_count,
    parse_weibo_datetime,
)
from ocd_v3.features.text import clean_weibo_text
from ocd_v3.io import write_json_atomic, write_jsonl_atomic

DATASET_SCHEMA_VERSION = 3


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_id(key: bytes) -> str:
    return hmac.new(key, b"ocd-v3/pseudonymization-key-id", hashlib.sha256).hexdigest()[:16]


def _pseudonym(key: bytes, namespace: str, *values: str) -> str:
    message = "\0".join((namespace, *values)).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()[:24]
    return f"{namespace}-{digest}"


def _dataset_id(config: StudyConfig, audit: RawAuditReport, key_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(DATASET_SCHEMA_VERSION).encode("ascii"))
    digest.update(config.digest().encode("ascii"))
    digest.update(audit.raw_inventory_sha256.encode("ascii"))
    digest.update(key_id.encode("ascii"))
    return f"dataset-{digest.hexdigest()[:20]}"


def _post_sort_key(post: dict[str, Any]) -> tuple[int, str, str]:
    timestamp = post.get("published_at")
    return (0 if timestamp is None else 1, timestamp or "", post["post_id"])


def _subject_rows(group_dir: Path) -> tuple[dict[str, dict[str, str]], set[str]]:
    _, rows = read_csv(group_dir / "users.csv")
    listed: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for row in rows:
        uid = str(row.get(USER_ID_COLUMN, "")).strip()
        if not uid:
            continue
        if uid in listed:
            duplicates.add(uid)
        listed[uid] = row
    if duplicates:
        raise RawDataError(
            f"Duplicate user IDs in {group_dir.name}/users.csv: count={len(duplicates)}"
        )
    return listed, {path.name for path in group_dir.iterdir() if path.is_dir()}


def _prepare_group(
    config: StudyConfig,
    group: GroupConfig,
    temporary_dir: Path,
    pseudonymization_key: bytes,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str], list[dict[str, Any]]]:
    group_dir = config.raw_root / group.directory
    listed, directory_ids = _subject_rows(group_dir)
    all_uids = sorted(set(listed) | directory_ids)
    subject_rows: list[dict[str, Any]] = []
    private_mapping_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    file_hashes: dict[str, str] = {}

    posts_dir = temporary_dir / "posts"
    media_dir = temporary_dir / "media"
    posts_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    for raw_uid in all_uids:
        subject_id = _pseudonym(pseudonymization_key, "subject", group.directory, raw_uid)
        subject_dir = group_dir / raw_uid
        post_csv = subject_dir / f"{raw_uid}.csv"
        exclusion_reason: str | None = None
        if raw_uid not in listed:
            exclusion_reason = "missing_user_metadata_row"
        elif not subject_dir.is_dir():
            exclusion_reason = "missing_user_directory"
        elif not post_csv.is_file():
            exclusion_reason = "missing_post_csv"

        posts: list[dict[str, Any]] = []
        media_rows: list[dict[str, Any]] = []
        seen_sids: set[str] = set()
        sid_to_post_id: dict[str, str] = {}
        private_posts: list[dict[str, str]] = []
        private_media: list[dict[str, str]] = []
        keyword_casefold = tuple(keyword.casefold() for keyword in config.dataset.keywords)

        if exclusion_reason is None:
            deduplicated_rows: dict[str, tuple[int, dict[str, str]]] = {}
            for row_number, row in iter_csv(post_csv):
                raw_sid = str(row.get(POST_ID_COLUMN, "")).strip()
                if not raw_sid:
                    counters["posts_missing_sid"] += 1
                    continue
                if raw_sid in deduplicated_rows:
                    counters["posts_duplicate_rows_dropped"] += 1
                    if deduplicated_rows[raw_sid][1] != row:
                        counters["posts_conflicting_duplicate_rows"] += 1
                deduplicated_rows[raw_sid] = (row_number, row)

            for raw_sid, (_row_number, row) in deduplicated_rows.items():
                seen_sids.add(raw_sid)
                post_id = _pseudonym(
                    pseudonymization_key, "post", group.directory, raw_uid, raw_sid
                )
                sid_to_post_id[raw_sid] = post_id
                private_posts.append({"post_id": post_id, "raw_post_id": raw_sid})
                cleaned_text = clean_weibo_text(
                    row.get(POST_TEXT_COLUMN), row.get(POST_LOCATION_COLUMN)
                )
                parsed_at = parse_weibo_datetime(
                    row.get(POST_DATE_COLUMN) or row.get(POST_DATE_FALLBACK_COLUMN)
                )
                likes = parse_nonnegative_count(row.get(LIKES_COLUMN))
                comments = parse_nonnegative_count(row.get(COMMENTS_COLUMN))
                reposts = parse_nonnegative_count(row.get(REPOSTS_COLUMN))
                posts.append(
                    {
                        "schema_version": 1,
                        "post_id": post_id,
                        "published_at": parsed_at.isoformat(sep=" ") if parsed_at else None,
                        "cleaned_text": cleaned_text,
                        "contains_study_keyword": any(
                            keyword in cleaned_text.casefold() for keyword in keyword_casefold
                        ),
                        "likes": likes,
                        "comments": comments,
                        "reposts": reposts,
                        "metadata_vector": list(
                            metadata_vector(likes, comments, reposts, parsed_at)
                        ),
                    }
                )
            posts.sort(key=_post_sort_key)
            if not posts:
                exclusion_reason = "no_valid_posts"

            for kind, media_path, raw_sid, ordinal in iter_subject_media(subject_dir):
                relative_path = media_path.relative_to(config.raw_root).as_posix()
                media_id = _pseudonym(
                    pseudonymization_key, "media", group.directory, raw_uid, relative_path
                )
                media_rows.append(
                    {
                        "schema_version": 1,
                        "media_id": media_id,
                        "post_id": sid_to_post_id.get(raw_sid or ""),
                        "kind": kind,
                        "ordinal": ordinal,
                        "extension": media_path.suffix.lower(),
                    }
                )
                private_media.append(
                    {"media_id": media_id, "source_relative_path": relative_path}
                )
                if raw_sid is None:
                    counters["media_unparseable"] += 1
                elif raw_sid not in sid_to_post_id:
                    counters["media_unmatched"] += 1

        posts_path = posts_dir / f"{subject_id}.jsonl"
        media_path = media_dir / f"{subject_id}.jsonl"
        write_jsonl_atomic(posts_path, posts)
        write_jsonl_atomic(media_path, media_rows)
        file_hashes[f"posts/{subject_id}.jsonl"] = sha256_file(posts_path)
        file_hashes[f"media/{subject_id}.jsonl"] = sha256_file(media_path)

        eligible = exclusion_reason is None
        counters["subjects_total"] += 1
        counters["subjects_eligible"] += int(eligible)
        counters["subjects_excluded"] += int(not eligible)
        counters["posts"] += len(posts)
        counters["media"] += len(media_rows)
        if exclusion_reason:
            counters[f"excluded_{exclusion_reason}"] += 1
        subject_rows.append(
            {
                "schema_version": 1,
                "subject_id": subject_id,
                "source_group": group.directory,
                "label_name": group.label_name,
                "label_id": group.label_id,
                "eligible": eligible,
                "exclusion_reason": exclusion_reason,
                "post_count": len(posts),
                "media_count": len(media_rows),
                "first_post_at": posts[0]["published_at"] if posts else None,
                "last_post_at": posts[-1]["published_at"] if posts else None,
            }
        )
        private_mapping_rows.append(
            {
                "schema_version": 1,
                "subject_id": subject_id,
                "raw_group_directory": group.directory,
                "raw_user_id": raw_uid,
                "posts": private_posts,
                "media": private_media,
            }
        )
    return subject_rows, dict(sorted(counters.items())), file_hashes, private_mapping_rows


def prepare_dataset(config: StudyConfig) -> tuple[Path, dict[str, Any]]:
    audit = audit_raw_dataset(config)
    pseudonymization_key = config.pseudonymization_key()
    pseudonymization_key_id = _key_id(pseudonymization_key)
    dataset_id = _dataset_id(config, audit, pseudonymization_key_id)
    final_dir = config.dataset_dir / dataset_id
    final_mapping_dir = mapping_directory(config, dataset_id)
    if final_dir.exists():
        manifest_path = final_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RawDataError(f"Incomplete dataset directory already exists: {dataset_id}")
        if not (final_mapping_dir / "identity-map.jsonl").is_file():
            raise RawDataError(
                "Prepared dataset exists but its required private identity mapping is missing"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return final_dir, manifest

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = config.dataset_dir / f".{dataset_id}.{uuid.uuid4().hex}.tmp"
    temporary_mapping_dir = (
        config.private_mapping_dir / f".{dataset_id}.{uuid.uuid4().hex}.tmp"
    )
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        all_subject_rows: list[dict[str, Any]] = []
        group_summaries: dict[str, dict[str, int]] = {}
        file_hashes: dict[str, str] = {}
        private_mapping_rows: list[dict[str, Any]] = []
        for group in config.dataset.groups:
            subject_rows, summary, group_hashes, group_private_mappings = _prepare_group(
                config, group, temporary_dir, pseudonymization_key
            )
            all_subject_rows.extend(subject_rows)
            group_summaries[group.label_name] = summary
            file_hashes.update(group_hashes)
            private_mapping_rows.extend(group_private_mappings)

        subjects_path = temporary_dir / "subjects.jsonl"
        write_jsonl_atomic(subjects_path, all_subject_rows)
        file_hashes["subjects.jsonl"] = sha256_file(subjects_path)
        file_index = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "files": file_hashes,
        }
        write_json_atomic(temporary_dir / "file_index.json", file_index)

        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_inventory_sha256": audit.raw_inventory_sha256,
            "study_config_sha256": config.digest(),
            "groups": group_summaries,
            "selection_rule": (
                "A subject is eligible only when its users.csv row, user directory, "
                "post CSV, and at least one valid post are all present."
            ),
            "post_order": "ascending by parsed publication time; missing times precede known times",
            "duplicate_post_policy": (
                "Within each subject, keep the last source row for a repeated raw post ID; "
                "record dropped and conflicting rows in the group summary."
            ),
            "post_selection_for_models": {
                "maximum_posts_per_subject": (
                    config.dataset.post_selection.maximum_posts_per_subject
                ),
                "strategy": config.dataset.post_selection.strategy,
            },
            "text_cleaning_version": 1,
            "pseudonymization": {
                "method": "HMAC-SHA256",
                "key_id": pseudonymization_key_id,
                "identifier_policy": "keyed pseudonymous subject, post, and media IDs",
                "private_mapping_location": "external to repository and artifact root",
            },
            "keyword_masking": {
                "condition_names": ["original", "masked"],
                "replacement": config.dataset.keyword_mask,
                "keyword_list_sha256": _sha256_json(list(config.dataset.keywords)),
            },
            "metadata_features": [
                "log1p_likes",
                "log1p_comments",
                "log1p_reposts",
                "posting_time_sin",
                "posting_time_cos",
            ],
            "identifier_policy": "No raw user, post, or media identifiers in this dataset",
        }
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if final_mapping_dir.exists():
            raise RawDataError(
                f"Private identity mapping directory already exists: {final_mapping_dir.name}"
            )
        write_private_mapping(temporary_mapping_dir, dataset_id, private_mapping_rows)
        final_mapping_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_mapping_dir, final_mapping_dir)
        os.replace(temporary_dir, final_dir)
        return final_dir, manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        shutil.rmtree(temporary_mapping_dir, ignore_errors=True)
        if final_mapping_dir.exists() and not final_dir.exists():
            shutil.rmtree(final_mapping_dir, ignore_errors=True)
        raise
