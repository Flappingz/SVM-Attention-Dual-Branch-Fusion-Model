from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, stdev


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class IntervalSummary:
    n: int
    mean: float
    sample_sd: float
    ci95_low: float
    ci95_high: float


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> BinaryMetrics:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not labels:
        raise ValueError("At least one prediction is required")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("Binary labels must be 0 or 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predictions = [int(score >= threshold) for score in scores]
    pairs = list(zip(predictions, labels, strict=True))
    tp = sum(prediction == 1 and label == 1 for prediction, label in pairs)
    tn = sum(prediction == 0 and label == 0 for prediction, label in pairs)
    fp = sum(prediction == 1 and label == 0 for prediction, label in pairs)
    fn = sum(prediction == 0 and label == 1 for prediction, label in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        n=len(labels),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=(tp + tn) / len(labels),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(labels, scores),
    )


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_sd_ci95(values: Iterable[float]) -> IntervalSummary:
    materialized = [float(value) for value in values]
    if len(materialized) < 2:
        raise ValueError("At least two independent fold values are required")
    mean = fmean(materialized)
    sample_sd = stdev(materialized)
    degrees_of_freedom = len(materialized) - 1
    critical = _T_CRITICAL_975.get(degrees_of_freedom, 1.96)
    margin = critical * sample_sd / math.sqrt(len(materialized))
    return IntervalSummary(
        n=len(materialized),
        mean=mean,
        sample_sd=sample_sd,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
    )
