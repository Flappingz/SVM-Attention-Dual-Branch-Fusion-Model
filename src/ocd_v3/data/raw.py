from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocd_v3.config import GroupConfig, StudyConfig
from ocd_v3.features.metadata import parse_nonnegative_count, parse_weibo_datetime

USER_ID_COLUMN = "用户id"
POST_ID_COLUMN = "id"
POST_TEXT_COLUMN = "正文"
POST_LOCATION_COLUMN = "位置"
POST_DATE_COLUMN = "完整日期"
POST_DATE_FALLBACK_COLUMN = "日期"
LIKES_COLUMN = "点赞数"
COMMENTS_COLUMN = "评论数"
REPOSTS_COLUMN = "转发数"

IMAGE_DIRECTORY_CANDIDATES = (Path("img/原创微博图片"), Path("image/原创微博图片"))
LIVE_PHOTO_DIRECTORY = Path("live_photo/原创微博Live Photo视频")


class RawDataError(RuntimeError):
    """Raised when the authoritative raw dataset violates a required invariant."""


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RawDataError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def read_csv_header(path: Path) -> list[str]:
    """Read a CSV schema without loading or traversing its data rows."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RawDataError(f"CSV has no header: {path}")
        return list(reader.fieldnames)


def iter_csv(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RawDataError(f"CSV has no header: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, dict(row)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_media_filename(path: Path) -> tuple[str | None, int | None]:
    parts = path.stem.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        return None, None
    ordinal = 1
    if len(parts) >= 3 and parts[-1].isdigit():
        ordinal = int(parts[-1])
    return parts[1], ordinal


def iter_subject_media(subject_dir: Path) -> Iterator[tuple[str, Path, str | None, int | None]]:
    seen: set[Path] = set()
    for relative_dir in IMAGE_DIRECTORY_CANDIDATES:
        media_dir = subject_dir / relative_dir
        if not media_dir.is_dir():
            continue
        with os.scandir(media_dir) as entries:
            paths = sorted(
                (Path(entry.path) for entry in entries if entry.is_file(follow_symlinks=False)),
                key=lambda item: item.name,
            )
        for path in paths:
            if path not in seen:
                seen.add(path)
                sid, ordinal = parse_media_filename(path)
                yield "image", path, sid, ordinal
    media_dir = subject_dir / LIVE_PHOTO_DIRECTORY
    if media_dir.is_dir():
        with os.scandir(media_dir) as entries:
            paths = sorted(
                (Path(entry.path) for entry in entries if entry.is_file(follow_symlinks=False)),
                key=lambda item: item.name,
            )
        for path in paths:
            if path not in seen:
                seen.add(path)
                sid, ordinal = parse_media_filename(path)
                yield "live_photo", path, sid, ordinal


@dataclass(frozen=True)
class GroupAudit:
    directory: str
    label_name: str
    label_id: int
    user_csv_rows: int
    unique_listed_users: int
    duplicate_user_rows: int
    user_directories: int
    listed_without_directory: int
    directory_without_listing: int
    missing_post_csv: int
    post_rows: int
    unique_post_ids: int
    duplicate_post_ids: int
    subjects_with_duplicate_post_ids: int
    conflicting_duplicate_post_ids: int
    post_ids_shared_between_subjects: int
    missing_post_ids: int
    invalid_dates: int
    earliest_post_at: str | None
    latest_post_at: str | None
    media_files: int
    image_files: int
    live_photo_files: int
    media_without_parseable_post_id: int
    media_without_matching_post: int
    media_extensions: dict[str, int]
    user_columns: list[str]
    post_columns: list[str]
    csv_content_sha256: str
    media_inventory_sha256: str


@dataclass(frozen=True)
class RawAuditReport:
    schema_version: int
    generated_at: str
    source: str
    config_sha256: str
    raw_inventory_sha256: str
    groups: tuple[GroupAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_group(raw_root: Path, group: GroupConfig) -> GroupAudit:
    group_dir = raw_root / group.directory
    users_csv = group_dir / "users.csv"
    if not group_dir.is_dir():
        raise RawDataError(f"Missing group directory: {group.directory}")
    if not users_csv.is_file():
        raise RawDataError(f"Missing users.csv for group: {group.directory}")

    user_columns, user_rows = read_csv(users_csv)
    if USER_ID_COLUMN not in user_columns:
        raise RawDataError(f"{users_csv} is missing {USER_ID_COLUMN}")
    listed_ids = [str(row.get(USER_ID_COLUMN, "")).strip() for row in user_rows]
    listed_ids = [value for value in listed_ids if value]
    listed_counter = Counter(listed_ids)
    listed_set = set(listed_ids)
    subject_dirs = sorted(
        (path for path in group_dir.iterdir() if path.is_dir()), key=lambda item: item.name
    )
    directory_ids = {path.name for path in subject_dirs}

    csv_digest = hashlib.sha256()

    def update_csv_digest(path: Path) -> None:
        csv_digest.update(path.relative_to(raw_root).as_posix().encode("utf-8"))
        csv_digest.update(b"\0")
        csv_digest.update(sha256_file(path).encode("ascii"))
        csv_digest.update(b"\0")

    update_csv_digest(users_csv)
    media_digest = hashlib.sha256()
    post_columns: list[str] = []
    seen_post_ids: set[str] = set()
    post_rows = 0
    duplicate_post_ids = 0
    subjects_with_duplicate_post_ids = 0
    conflicting_duplicate_post_ids = 0
    post_ids_shared_between_subjects = 0
    missing_post_ids = 0
    missing_post_csv = 0
    invalid_dates = 0
    timestamps: list[datetime] = []
    media_files = 0
    image_files = 0
    live_photo_files = 0
    media_unparsed = 0
    media_unmatched = 0
    extensions: Counter[str] = Counter()

    for subject_dir in subject_dirs:
        post_csv = subject_dir / f"{subject_dir.name}.csv"
        subject_post_ids: set[str] = set()
        subject_post_rows: dict[str, str] = {}
        subject_conflicting_ids: set[str] = set()
        subject_has_duplicates = False
        if not post_csv.is_file():
            missing_post_csv += 1
        else:
            columns = read_csv_header(post_csv)
            if not post_columns:
                post_columns = columns
            update_csv_digest(post_csv)
            for _, row in iter_csv(post_csv):
                post_rows += 1
                sid = str(row.get(POST_ID_COLUMN, "")).strip()
                if not sid:
                    missing_post_ids += 1
                else:
                    canonical_row = json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    if sid in subject_post_rows:
                        duplicate_post_ids += 1
                        subject_has_duplicates = True
                        if subject_post_rows[sid] != canonical_row:
                            subject_conflicting_ids.add(sid)
                    elif sid in seen_post_ids:
                        duplicate_post_ids += 1
                        post_ids_shared_between_subjects += 1
                    seen_post_ids.add(sid)
                    subject_post_ids.add(sid)
                    subject_post_rows[sid] = canonical_row
                raw_date = row.get(POST_DATE_COLUMN) or row.get(POST_DATE_FALLBACK_COLUMN)
                parsed_date = parse_weibo_datetime(raw_date)
                if parsed_date is None and str(raw_date or "").strip():
                    invalid_dates += 1
                elif parsed_date is not None:
                    timestamps.append(parsed_date)
                parse_nonnegative_count(row.get(LIKES_COLUMN))
                parse_nonnegative_count(row.get(COMMENTS_COLUMN))
                parse_nonnegative_count(row.get(REPOSTS_COLUMN))
            subjects_with_duplicate_post_ids += int(subject_has_duplicates)
            conflicting_duplicate_post_ids += len(subject_conflicting_ids)

        for kind, media_path, sid, _ in iter_subject_media(subject_dir):
            media_files += 1
            image_files += int(kind == "image")
            live_photo_files += int(kind == "live_photo")
            extensions[media_path.suffix.lower() or "<none>"] += 1
            if sid is None:
                media_unparsed += 1
            elif sid not in subject_post_ids:
                media_unmatched += 1
            relative = media_path.relative_to(raw_root).as_posix()
            media_digest.update(relative.encode("utf-8"))
            media_digest.update(b"\0")

    return GroupAudit(
        directory=group.directory,
        label_name=group.label_name,
        label_id=group.label_id,
        user_csv_rows=len(user_rows),
        unique_listed_users=len(listed_set),
        duplicate_user_rows=sum(count - 1 for count in listed_counter.values()),
        user_directories=len(subject_dirs),
        listed_without_directory=len(listed_set - directory_ids),
        directory_without_listing=len(directory_ids - listed_set),
        missing_post_csv=missing_post_csv,
        post_rows=post_rows,
        unique_post_ids=len(seen_post_ids),
        duplicate_post_ids=duplicate_post_ids,
        subjects_with_duplicate_post_ids=subjects_with_duplicate_post_ids,
        conflicting_duplicate_post_ids=conflicting_duplicate_post_ids,
        post_ids_shared_between_subjects=post_ids_shared_between_subjects,
        missing_post_ids=missing_post_ids,
        invalid_dates=invalid_dates,
        earliest_post_at=min(timestamps).isoformat(sep=" ") if timestamps else None,
        latest_post_at=max(timestamps).isoformat(sep=" ") if timestamps else None,
        media_files=media_files,
        image_files=image_files,
        live_photo_files=live_photo_files,
        media_without_parseable_post_id=media_unparsed,
        media_without_matching_post=media_unmatched,
        media_extensions=dict(sorted(extensions.items())),
        user_columns=user_columns,
        post_columns=post_columns,
        csv_content_sha256=csv_digest.hexdigest(),
        media_inventory_sha256=media_digest.hexdigest(),
    )


def audit_raw_dataset(config: StudyConfig) -> RawAuditReport:
    if not config.raw_root.is_dir():
        raise RawDataError("Configured raw_root does not exist")
    groups = tuple(_audit_group(config.raw_root, group) for group in config.dataset.groups)
    inventory_digest = hashlib.sha256()
    for group in groups:
        inventory_digest.update(group.csv_content_sha256.encode("ascii"))
        inventory_digest.update(group.media_inventory_sha256.encode("ascii"))
    return RawAuditReport(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source="configured authoritative raw directory",
        config_sha256=config.digest(),
        raw_inventory_sha256=inventory_digest.hexdigest(),
        groups=groups,
    )
