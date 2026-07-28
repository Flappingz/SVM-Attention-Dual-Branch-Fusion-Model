"""Strict configuration for the frozen Qwen3-VL mean-pooling MLP baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocd_v3.experiments.full_config import (
    TrainingConfig,
    parse_hierarchical_training_config,
)


@dataclass(frozen=True)
class QwenVLMeanMLPEncoderConfig:
    model_id: str
    revision: str
    representations: tuple[str, ...]
    normalize: bool
    device: str
    frozen: bool
    metadata: str


@dataclass(frozen=True)
class QwenVLMeanMLPModelConfig:
    post_pooling: str
    input_normalization: str
    hidden_dimension: int
    activation: str
    dropout: float


@dataclass(frozen=True)
class QwenVLMeanMLPConfig:
    schema_version: int
    encoder: QwenVLMeanMLPEncoderConfig
    model: QwenVLMeanMLPModelConfig
    training: TrainingConfig

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder": asdict(self.encoder),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} key(s): {', '.join(unknown)}")


def load_qwen_vl_mean_mlp_config(path: str | Path) -> QwenVLMeanMLPConfig:
    payload = _mapping(json.loads(Path(path).read_text(encoding="utf-8")), "configuration")
    _reject_unknown(payload, {"schema_version", "encoder", "model", "training"}, "top-level")
    if payload.get("schema_version") != 1:
        raise ValueError("Only Qwen3-VL mean-MLP configuration schema_version=1 is supported")

    encoder_payload = _mapping(payload.get("encoder"), "encoder")
    _reject_unknown(
        encoder_payload,
        {
            "model_id",
            "revision",
            "representations",
            "normalize",
            "device",
            "frozen",
            "metadata",
        },
        "encoder",
    )
    raw_representations = encoder_payload.get("representations")
    if raw_representations != ["final"]:
        raise ValueError("Qwen3-VL mean pooling requires representations=['final']")
    encoder = QwenVLMeanMLPEncoderConfig(
        model_id=str(encoder_payload["model_id"]).strip(),
        revision=str(encoder_payload["revision"]).strip(),
        representations=("final",),
        normalize=bool(encoder_payload["normalize"]),
        device=str(encoder_payload["device"]),
        frozen=bool(encoder_payload["frozen"]),
        metadata=str(encoder_payload["metadata"]),
    )
    if not encoder.model_id or not encoder.revision:
        raise ValueError("encoder.model_id and encoder.revision must be non-empty")
    if encoder.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("encoder.device must be cpu, cuda, or auto")
    if not encoder.frozen:
        raise ValueError("The Qwen3-VL embedding encoder must remain frozen")
    if not encoder.normalize:
        raise ValueError("The fixed baseline requires normalized Qwen3-VL embeddings")
    if encoder.metadata != "excluded":
        raise ValueError("This baseline requires metadata='excluded'")

    model_payload = _mapping(payload.get("model"), "model")
    _reject_unknown(
        model_payload,
        {
            "post_pooling",
            "input_normalization",
            "hidden_dimension",
            "activation",
            "dropout",
        },
        "model",
    )
    model = QwenVLMeanMLPModelConfig(
        post_pooling=str(model_payload["post_pooling"]),
        input_normalization=str(model_payload["input_normalization"]),
        hidden_dimension=int(model_payload["hidden_dimension"]),
        activation=str(model_payload["activation"]),
        dropout=float(model_payload["dropout"]),
    )
    if model.post_pooling != "masked_mean":
        raise ValueError("model.post_pooling must be masked_mean")
    if model.input_normalization != "layer_norm":
        raise ValueError("model.input_normalization must be layer_norm")
    if model.hidden_dimension < 1:
        raise ValueError("model.hidden_dimension must be positive")
    if model.activation != "gelu":
        raise ValueError("model.activation must be gelu")
    if not 0 <= model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")

    return QwenVLMeanMLPConfig(
        schema_version=1,
        encoder=encoder,
        model=model,
        training=parse_hierarchical_training_config(payload.get("training")),
    )
