"""Strict protocol configuration for the four-model controlled benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTROLLED_MODEL_IDS = (
    "tfidf_linear_svm",
    "chinese_roberta_mean_logistic",
    "qwen_vl_mean_mlp",
    "hierarchical_attention_full",
)
CONTROLLED_METRICS = ("f1", "roc_auc", "accuracy", "precision", "recall")


@dataclass(frozen=True)
class ControlledCohortConfig:
    field: str
    operator: str
    threshold: int

    @property
    def minimum_posts_per_subject(self) -> int:
        """Translate the strict >N rule to the split API's inclusive minimum."""
        return self.threshold + 1


@dataclass(frozen=True)
class ControlledSplitConfig:
    seeds: tuple[int, ...]
    outer_folds: int
    validation_fraction_within_outer_train: float


@dataclass(frozen=True)
class ControlledBenchmarkConfig:
    schema_version: int
    expected_dataset_id: str | None
    keyword_condition: str
    cohort: ControlledCohortConfig
    splits: ControlledSplitConfig
    training_base_seed: int
    classification_threshold: float
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    models: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        # Return the exact JSON representation used in manifests (tuples become lists).
        return json.loads(json.dumps(asdict(self)))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} key(s): {', '.join(unknown)}")


def load_controlled_benchmark_config(path: str | Path) -> ControlledBenchmarkConfig:
    payload = _mapping(json.loads(Path(path).read_text(encoding="utf-8")), "configuration")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "expected_dataset_id",
            "keyword_condition",
            "cohort",
            "splits",
            "training_base_seed",
            "classification_threshold",
            "primary_metric",
            "secondary_metrics",
            "models",
        },
        "top-level",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Only controlled benchmark schema_version=1 is supported")

    cohort_payload = _mapping(payload.get("cohort"), "cohort")
    _reject_unknown(cohort_payload, {"field", "operator", "threshold"}, "cohort")
    cohort = ControlledCohortConfig(
        field=str(cohort_payload["field"]),
        operator=str(cohort_payload["operator"]),
        threshold=int(cohort_payload["threshold"]),
    )
    if cohort != ControlledCohortConfig(
        field="available_post_count", operator=">", threshold=20
    ):
        raise ValueError(
            "The controlled benchmark cohort must be available_post_count > 20"
        )

    split_payload = _mapping(payload.get("splits"), "splits")
    _reject_unknown(
        split_payload,
        {"seeds", "outer_folds", "validation_fraction_within_outer_train"},
        "splits",
    )
    raw_seeds = split_payload.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("splits.seeds must be a list")
    splits = ControlledSplitConfig(
        seeds=tuple(int(value) for value in raw_seeds),
        outer_folds=int(split_payload["outer_folds"]),
        validation_fraction_within_outer_train=float(
            split_payload["validation_fraction_within_outer_train"]
        ),
    )
    if len(splits.seeds) != 8 or len(set(splits.seeds)) != 8:
        raise ValueError("The controlled benchmark requires exactly eight unique split seeds")
    if splits.outer_folds != 5:
        raise ValueError("The controlled benchmark requires exactly five outer folds")
    if not 0 < splits.validation_fraction_within_outer_train < 1:
        raise ValueError("The validation fraction must be between zero and one")

    models = tuple(str(value) for value in payload.get("models", []))
    if models != CONTROLLED_MODEL_IDS:
        raise ValueError(
            "models must contain only the four controlled experiments in canonical order"
        )
    primary_metric = str(payload["primary_metric"])
    secondary_metrics = tuple(str(value) for value in payload.get("secondary_metrics", []))
    if primary_metric != "f1" or secondary_metrics != CONTROLLED_METRICS[1:]:
        raise ValueError("Metrics must be F1 primary followed by the fixed secondary metrics")
    if str(payload["keyword_condition"]) != "removed":
        raise ValueError("The controlled benchmark requires Keyword-post-removed")
    threshold = float(payload["classification_threshold"])
    if threshold != 0.5:
        raise ValueError("The controlled benchmark classification threshold must be 0.5")

    raw_dataset_id = payload.get("expected_dataset_id")
    if raw_dataset_id is None:
        dataset_id = None
    elif isinstance(raw_dataset_id, str):
        dataset_id = raw_dataset_id.strip()
        if not dataset_id:
            raise ValueError("expected_dataset_id must be non-empty when provided")
    else:
        raise ValueError("expected_dataset_id must be a string or null")

    return ControlledBenchmarkConfig(
        schema_version=1,
        expected_dataset_id=dataset_id,
        keyword_condition="removed",
        cohort=cohort,
        splits=splits,
        training_base_seed=int(payload["training_base_seed"]),
        classification_threshold=threshold,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        models=models,
    )
