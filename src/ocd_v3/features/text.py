from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_SHARE_MEDIA_ONLY = re.compile(r"^\s*分享(?:图片|视频)\s*$")
_VIDEO_SUFFIX = re.compile(r"\s*\n[^\n\r]{1,120}的微博视频\s*$")


def clean_weibo_text(text: object, location: object = None) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if _SHARE_MEDIA_ONLY.fullmatch(value):
        return ""

    cleaned = _VIDEO_SUFFIX.sub("", value)
    normalized_location = "" if location is None else str(location).strip()
    if normalized_location:
        lines = cleaned.split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip() == normalized_location:
            lines.pop()
        cleaned = "\n".join(lines)

    compact_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if compact_lines and compact_lines[0] in {"分享图片", "分享视频"}:
        if len(compact_lines) <= 2:
            return ""
    return cleaned.rstrip()


@lru_cache(maxsize=32)
def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(set(keywords), key=lambda value: (-len(value), value.casefold()))
    return re.compile("|".join(re.escape(value) for value in ordered), re.IGNORECASE)


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Return whether text contains any configured study keyword.

    Matching deliberately uses the same case-insensitive substring semantics as
    ``mask_keywords`` so that keyword masking and post exclusion have one
    auditable definition.
    """
    normalized = tuple(str(value).strip() for value in keywords if str(value).strip())
    return bool(text and normalized and _keyword_pattern(normalized).search(text))


def count_keyword_matches(text: str, keywords: Iterable[str]) -> int:
    """Count non-overlapping matches using the same rule as ``mask_keywords``."""
    normalized = tuple(str(value).strip() for value in keywords if str(value).strip())
    if not text or not normalized:
        return 0
    return len(_keyword_pattern(normalized).findall(text))


def mask_keywords(text: str, keywords: Iterable[str], replacement: str = "[MASK]") -> str:
    normalized = tuple(str(value).strip() for value in keywords if str(value).strip())
    if not normalized or not text:
        return text
    return _keyword_pattern(normalized).sub(replacement, text)
