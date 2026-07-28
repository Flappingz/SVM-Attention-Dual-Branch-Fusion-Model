from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ocd_v3.evaluation.metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class OOFPrediction:
    subject_id: str
    outer_fold: int
    label: int
    score: float


def evaluate_oof(
    predictions: Iterable[OOFPrediction], expected_subject_ids: set[str], threshold: float
) -> BinaryMetrics:
    rows = list(predictions)
    counts = Counter(row.subject_id for row in rows)
    if set(counts) != expected_subject_ids:
        raise ValueError("OOF predictions do not cover the expected subject set")
    if any(count != 1 for count in counts.values()):
        raise ValueError("Each subject must have exactly one OOF prediction")
    ordered = sorted(rows, key=lambda row: row.subject_id)
    return binary_metrics(
        [row.label for row in ordered], [row.score for row in ordered], threshold=threshold
    )
