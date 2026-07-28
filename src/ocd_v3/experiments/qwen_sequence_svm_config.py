"""Strict config for the full-width Qwen sequence-moments Linear SVM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QwenSequenceEncoderConfig:
    model_id: str
    revision: str
    representation: str
    embedding_dimension: int
    normalize: bool
    frozen: bool


@dataclass(frozen=True)
class QwenSequencePoolingConfig:
    method: str
    chronological_blocks: tuple[str, ...]
    block_normalization: str
    final_normalization: str
    embedding_prefix_dimension: int | None = None


@dataclass(frozen=True)
class QwenSequenceSVMClassifierConfig:
    estimator: str
    c: float
    loss: str
    class_weight: str | None
    tolerance: float
    maximum_iterations: int
    random_state: int


@dataclass(frozen=True)
class QwenSequenceSVMConfig:
    schema_version: int
    encoder: QwenSequenceEncoderConfig
    sequence_pooling: QwenSequencePoolingConfig
    classifier: QwenSequenceSVMClassifierConfig

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.sequence_pooling.embedding_prefix_dimension is None:
            payload["sequence_pooling"].pop("embedding_prefix_dimension")
        return payload


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} key(s): {', '.join(unknown)}")


def load_qwen_sequence_svm_config(path: str | Path) -> QwenSequenceSVMConfig:
    payload = _mapping(json.loads(Path(path).read_text(encoding="utf-8")), "configuration")
    _reject_unknown(
        payload,
        {"schema_version", "encoder", "sequence_pooling", "classifier"},
        "top-level",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Only Qwen sequence SVM schema_version=1 is supported")

    raw_encoder = _mapping(payload.get("encoder"), "encoder")
    _reject_unknown(
        raw_encoder,
        {
            "model_id",
            "revision",
            "representation",
            "embedding_dimension",
            "normalize",
            "frozen",
        },
        "encoder",
    )
    encoder = QwenSequenceEncoderConfig(
        model_id=str(raw_encoder["model_id"]),
        revision=str(raw_encoder["revision"]),
        representation=str(raw_encoder["representation"]),
        embedding_dimension=int(raw_encoder["embedding_dimension"]),
        normalize=bool(raw_encoder["normalize"]),
        frozen=bool(raw_encoder["frozen"]),
    )
    if encoder.model_id != "Qwen/Qwen3-VL-Embedding-2B":
        raise ValueError("encoder.model_id must be Qwen/Qwen3-VL-Embedding-2B")
    if not encoder.revision:
        raise ValueError("encoder.revision must be non-empty")
    if encoder.representation != "final" or encoder.embedding_dimension != 2048:
        raise ValueError("The locked model requires the full 2048-dimensional final embedding")
    if not encoder.normalize or not encoder.frozen:
        raise ValueError("The locked Qwen embeddings must be normalized and frozen")

    raw_pooling = _mapping(payload.get("sequence_pooling"), "sequence_pooling")
    _reject_unknown(
        raw_pooling,
        {
            "method",
            "chronological_blocks",
            "block_normalization",
            "final_normalization",
            "embedding_prefix_dimension",
        },
        "sequence_pooling",
    )
    raw_blocks = raw_pooling.get("chronological_blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("sequence_pooling.chronological_blocks must be a list")
    pooling = QwenSequencePoolingConfig(
        method=str(raw_pooling["method"]),
        chronological_blocks=tuple(str(value) for value in raw_blocks),
        block_normalization=str(raw_pooling["block_normalization"]),
        final_normalization=str(raw_pooling["final_normalization"]),
        embedding_prefix_dimension=(
            int(raw_pooling["embedding_prefix_dimension"])
            if raw_pooling.get("embedding_prefix_dimension") is not None
            else None
        ),
    )
    expected_blocks_by_method = {
        "mean_std_temporal_halves": (
            "global_mean",
            "global_std",
            "early_half_mean",
            "late_half_mean",
        ),
        "temporal_pyramid2": (
            "global_mean",
            "early_half_mean",
            "late_half_mean",
        ),
        "segment_mean_std2": (
            "global_mean",
            "global_std",
            "early_half_mean",
            "late_half_mean",
            "early_half_std",
            "late_half_std",
        ),
    }
    if pooling.method not in expected_blocks_by_method:
        raise ValueError("Unknown locked Qwen sequence pooling method")
    if pooling.chronological_blocks != expected_blocks_by_method[pooling.method]:
        raise ValueError("The chronological sequence blocks differ from the method")
    if pooling.block_normalization != "l2" or pooling.final_normalization != "l2":
        raise ValueError("The locked model requires block and final L2 normalization")
    if (
        pooling.embedding_prefix_dimension is not None
        and not 1
        <= pooling.embedding_prefix_dimension
        <= encoder.embedding_dimension
    ):
        raise ValueError(
            "sequence_pooling.embedding_prefix_dimension must be within the encoder width"
        )

    raw_classifier = _mapping(payload.get("classifier"), "classifier")
    _reject_unknown(
        raw_classifier,
        {
            "estimator",
            "c",
            "loss",
            "class_weight",
            "tolerance",
            "maximum_iterations",
            "random_state",
        },
        "classifier",
    )
    classifier = QwenSequenceSVMClassifierConfig(
        estimator=str(raw_classifier["estimator"]),
        c=float(raw_classifier["c"]),
        loss=str(raw_classifier["loss"]),
        class_weight=(
            str(raw_classifier["class_weight"])
            if raw_classifier.get("class_weight") is not None
            else None
        ),
        tolerance=float(raw_classifier["tolerance"]),
        maximum_iterations=int(raw_classifier["maximum_iterations"]),
        random_state=int(raw_classifier["random_state"]),
    )
    if classifier.estimator != "linear_svc":
        raise ValueError("classifier.estimator must be linear_svc")
    expected_c = (
        0.1
        if pooling.method in {"mean_std_temporal_halves", "segment_mean_std2"}
        else 1.0
    )
    if classifier.c != expected_c or classifier.loss != "squared_hinge":
        raise ValueError(
            f"The locked {pooling.method} classifier requires C={expected_c:g} "
            "and squared_hinge"
        )
    if classifier.class_weight is not None:
        raise ValueError("The locked classifier uses no class weighting")
    if classifier.tolerance <= 0 or classifier.maximum_iterations < 1:
        raise ValueError("Classifier tolerance/iterations are invalid")

    return QwenSequenceSVMConfig(
        schema_version=1,
        encoder=encoder,
        sequence_pooling=pooling,
        classifier=classifier,
    )
