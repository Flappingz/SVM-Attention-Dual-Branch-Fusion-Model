from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.features.text import contains_keywords, mask_keywords
from ocd_v3.io import read_jsonl

KEYWORD_CONDITIONS = frozenset({"original", "masked", "removed"})


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label_name: str
    label_id: int
    post_count: int
    media_count: int
    selected_post_count: int | None = None

    @property
    def split_post_count(self) -> int:
        return self.post_count if self.selected_post_count is None else self.selected_post_count


class PreparedDataset:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.manifest = json.loads(
            (self.dataset_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self._subject_rows = read_jsonl(self.dataset_dir / "subjects.jsonl")

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    def subjects(self, *, eligible_only: bool = True) -> Iterator[SubjectRecord]:
        maximum_posts = int(
            self.manifest["post_selection_for_models"]["maximum_posts_per_subject"]
        )
        for row in self._subject_rows:
            if eligible_only and not row["eligible"]:
                continue
            yield SubjectRecord(
                subject_id=str(row["subject_id"]),
                label_name=str(row["label_name"]),
                label_id=int(row["label_id"]),
                post_count=int(row["post_count"]),
                media_count=int(row["media_count"]),
                selected_post_count=min(int(row["post_count"]), maximum_posts),
            )

    def posts(
        self,
        subject_id: str,
        *,
        maximum_posts: int | None = None,
        keyword_condition: str = "original",
        keywords: tuple[str, ...] = (),
        replacement: str = "[MASK]",
    ) -> list[dict[str, Any]]:
        if keyword_condition not in KEYWORD_CONDITIONS:
            raise ValueError("keyword_condition must be original, masked, or removed")
        rows = read_jsonl(self.dataset_dir / "posts" / f"{subject_id}.jsonl")
        if maximum_posts is not None:
            if maximum_posts < 1:
                raise ValueError("maximum_posts must be positive")
            rows = rows[-maximum_posts:]
        if keyword_condition == "removed":
            # The deterministic model-input window is fixed before any
            # condition is applied.  This keeps the condition on the same
            # source posts as original/masked and never fills gaps with older
            # posts from a different temporal window.
            rows = [
                row
                for row in rows
                if not contains_keywords(str(row["cleaned_text"]), keywords)
            ]
        if keyword_condition == "masked":
            rows = [
                {
                    **row,
                    "cleaned_text": mask_keywords(
                        str(row["cleaned_text"]), keywords, replacement
                    ),
                }
                for row in rows
            ]
        return rows

    def media(self, subject_id: str) -> list[dict[str, Any]]:
        return read_jsonl(self.dataset_dir / "media" / f"{subject_id}.jsonl")

    def selected_posts(
        self,
        subject_id: str,
        *,
        keyword_condition: str = "original",
        keywords: tuple[str, ...] = (),
        replacement: str = "[MASK]",
    ) -> list[dict[str, Any]]:
        maximum_posts = int(
            self.manifest["post_selection_for_models"]["maximum_posts_per_subject"]
        )
        return self.posts(
            subject_id,
            maximum_posts=maximum_posts,
            keyword_condition=keyword_condition,
            keywords=keywords,
            replacement=replacement,
        )
