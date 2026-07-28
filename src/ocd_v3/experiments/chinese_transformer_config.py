"""Strict configuration for the frozen Chinese Transformer baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChineseTransformerEncoderConfig:
    model_id: str
    revision: str
    maximum_tokens_per_post: int
    inference_batch_size: int
    device: str
    inference_dtype: str
    token_pooling: str
    post_pooling: str
    frozen: bool

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChineseTransformerClassifierConfig:
    standardize: bool
    classifier: str
    penalty: str
    c: float
    solver: str
    maximum_iterations: int
    class_weight: str | None
    random_state: int

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChineseTransformerBaselineConfig:
    schema_version: int
    encoder: ChineseTransformerEncoderConfig
    classifier: ChineseTransformerClassifierConfig

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder": self.encoder.public_dict(),
            "classifier": self.classifier.public_dict(),
        }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} key(s): {', '.join(unknown)}")


def load_chinese_transformer_config(path: str | Path) -> ChineseTransformerBaselineConfig:
    payload = _mapping(json.loads(Path(path).read_text(encoding="utf-8")), "configuration")
    _reject_unknown(payload, {"schema_version", "encoder", "classifier"}, "top-level")
    if payload.get("schema_version") != 1:
        raise ValueError("Only Chinese Transformer configuration schema_version=1 is supported")

    encoder_payload = _mapping(payload.get("encoder"), "encoder")
    _reject_unknown(
        encoder_payload,
        {
            "model_id",
            "revision",
            "maximum_tokens_per_post",
            "inference_batch_size",
            "device",
            "inference_dtype",
            "token_pooling",
            "post_pooling",
            "frozen",
        },
        "encoder",
    )
    encoder = ChineseTransformerEncoderConfig(
        model_id=str(encoder_payload["model_id"]),
        revision=str(encoder_payload["revision"]),
        maximum_tokens_per_post=int(encoder_payload["maximum_tokens_per_post"]),
        inference_batch_size=int(encoder_payload["inference_batch_size"]),
        device=str(encoder_payload["device"]),
        inference_dtype=str(encoder_payload["inference_dtype"]),
        token_pooling=str(encoder_payload["token_pooling"]),
        post_pooling=str(encoder_payload["post_pooling"]),
        frozen=bool(encoder_payload["frozen"]),
    )
    if not encoder.model_id or not encoder.revision:
        raise ValueError("encoder.model_id and encoder.revision must be non-empty")
    if encoder.maximum_tokens_per_post < 3:
        raise ValueError("encoder.maximum_tokens_per_post must be at least 3")
    if encoder.inference_batch_size < 1:
        raise ValueError("encoder.inference_batch_size must be positive")
    if encoder.device not in {"cpu", "cuda"}:
        raise ValueError("encoder.device must be cpu or cuda")
    if encoder.inference_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Unsupported encoder.inference_dtype")
    if encoder.device == "cpu" and encoder.inference_dtype == "float16":
        raise ValueError("float16 inference is not supported on CPU")
    if encoder.token_pooling != "non_special_token_mean":
        raise ValueError("Only non_special_token_mean token pooling is supported")
    if encoder.post_pooling != "mean":
        raise ValueError("Only mean post pooling is supported")
    if not encoder.frozen:
        raise ValueError("This baseline requires a frozen encoder")

    classifier_payload = _mapping(payload.get("classifier"), "classifier")
    _reject_unknown(
        classifier_payload,
        {
            "standardize",
            "classifier",
            "penalty",
            "c",
            "solver",
            "maximum_iterations",
            "class_weight",
            "random_state",
        },
        "classifier",
    )
    raw_class_weight = classifier_payload.get("class_weight")
    classifier = ChineseTransformerClassifierConfig(
        standardize=bool(classifier_payload["standardize"]),
        classifier=str(classifier_payload["classifier"]),
        penalty=str(classifier_payload["penalty"]),
        c=float(classifier_payload["c"]),
        solver=str(classifier_payload["solver"]),
        maximum_iterations=int(classifier_payload["maximum_iterations"]),
        class_weight=None if raw_class_weight is None else str(raw_class_weight),
        random_state=int(classifier_payload["random_state"]),
    )
    if classifier.classifier != "logistic_regression":
        raise ValueError("Only logistic_regression is supported")
    if not classifier.standardize:
        raise ValueError("The fixed baseline requires train-only feature standardization")
    if classifier.penalty != "l2" or classifier.solver != "liblinear":
        raise ValueError("The fixed baseline requires l2 penalty with liblinear")
    if classifier.c <= 0 or classifier.maximum_iterations < 1:
        raise ValueError("classifier.c and maximum_iterations must be positive")
    if classifier.class_weight not in {None, "balanced"}:
        raise ValueError("classifier.class_weight must be null or balanced")

    return ChineseTransformerBaselineConfig(
        schema_version=1,
        encoder=encoder,
        classifier=classifier,
    )
