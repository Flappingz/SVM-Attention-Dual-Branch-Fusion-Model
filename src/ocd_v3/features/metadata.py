from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean, pstdev

METADATA_FEATURE_NAMES = (
    "log1p_likes",
    "log1p_comments",
    "log1p_reposts",
    "posting_time_sin",
    "posting_time_cos",
)


def parse_nonnegative_count(value: object) -> int:
    try:
        parsed = int(float(str(value).strip() or "0"))
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def parse_weibo_datetime(value: object) -> datetime | None:
    token = "" if value is None else str(value).strip()
    if not token:
        return None
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
    )
    for date_format in formats:
        try:
            return datetime.strptime(token, date_format)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def metadata_vector(
    likes: int, comments: int, reposts: int, published_at: datetime | None
) -> tuple[float, float, float, float, float]:
    if published_at is None:
        time_sin = 0.0
        time_cos = 0.0
    else:
        hour = published_at.hour + published_at.minute / 60 + published_at.second / 3600
        angle = 2 * math.pi * hour / 24
        time_sin = math.sin(angle)
        time_cos = math.cos(angle)
    return (
        math.log1p(max(likes, 0)),
        math.log1p(max(comments, 0)),
        math.log1p(max(reposts, 0)),
        time_sin,
        time_cos,
    )


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Iterable[Sequence[float]]) -> TrainOnlyStandardizer:
        materialized = [tuple(float(value) for value in row) for row in rows]
        if not materialized:
            raise ValueError("Cannot fit a standardizer without training rows")
        width = len(materialized[0])
        if any(len(row) != width for row in materialized):
            raise ValueError("All metadata rows must have the same width")
        if width == 0:
            return cls(mean=(), scale=())
        columns = list(zip(*materialized, strict=True))
        means = tuple(fmean(column) for column in columns)
        scales = tuple(pstdev(column) or 1.0 for column in columns)
        return cls(mean=means, scale=scales)

    def transform(self, row: Sequence[float]) -> tuple[float, ...]:
        if len(row) != len(self.mean):
            raise ValueError("Metadata width differs from the fitted training schema")
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(row, self.mean, self.scale, strict=True)
        )
