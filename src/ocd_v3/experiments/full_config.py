from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class FullExperimentConfigurationError(ValueError):
    """Raised when the hierarchical full-experiment configuration is invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REPRESENTATION_PATTERN = re.compile(r"layer:([1-9][0-9]*)")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                missing.add(name)
                return match.group(0)
            return os.environ[name]

        expanded = _ENV_PATTERN.sub(replace, value)
        if missing:
            raise FullExperimentConfigurationError(
                "Missing required environment variable(s): " + ", ".join(sorted(missing))
            )
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullExperimentConfigurationError(f"{name} must be an object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FullExperimentConfigurationError(
            f"Unknown {name} key(s): {', '.join(unknown)}"
        )


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str
    model_name_or_path: Path
    revision: str
    representations: tuple[str, ...]
    normalize: bool
    instruction: str
    torch_dtype: str
    device: str
    max_length: int
    min_pixels: int
    max_pixels: int
    total_pixels: int
    fps: float
    max_frames: int
    max_images_per_post: int
    max_dynamic_media_per_post: int


@dataclass(frozen=True)
class HierarchicalModelConfig:
    d_model: int
    post_attention_heads: int
    user_attention_heads: int
    user_position_encoding: str
    ffn_hidden_dim: int
    classifier_hidden_dim: int
    dropout: float
    user_pooling: str
    embedding_prefix_dim: int | None = None
    relative_position_max_distance: int | None = None
    attention_inner_dim: int | None = None
    raw_mean_skip_dim: int | None = None
    classifier_head: str = "mlp"
    architecture_variant: str = "full"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    selection_metric: str
    seed: int
    num_workers: int


@dataclass(frozen=True)
class FullExperimentConfig:
    schema_version: int
    encoder: EncoderConfig
    model: HierarchicalModelConfig
    training: TrainingConfig

    def public_dict(self) -> dict[str, Any]:
        """Return a manifest-safe configuration without local filesystem paths."""
        payload = asdict(self)
        payload["encoder"].pop("model_name_or_path", None)
        if payload["model"].get("embedding_prefix_dim") is None:
            payload["model"].pop("embedding_prefix_dim", None)
        if payload["model"].get("relative_position_max_distance") is None:
            payload["model"].pop("relative_position_max_distance", None)
        if payload["model"].get("attention_inner_dim") is None:
            payload["model"].pop("attention_inner_dim", None)
        if payload["model"].get("raw_mean_skip_dim") is None:
            payload["model"].pop("raw_mean_skip_dim", None)
        if payload["model"].get("classifier_head") == "mlp":
            payload["model"].pop("classifier_head", None)
        if payload["model"].get("architecture_variant") == "full":
            payload["model"].pop("architecture_variant", None)
        return payload

def _parse_encoder(payload: Any) -> EncoderConfig:
    value = _mapping(payload, "encoder")
    allowed = {
        "model_id",
        "model_name_or_path",
        "revision",
        "representations",
        "normalize",
        "instruction",
        "torch_dtype",
        "device",
        "max_length",
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "fps",
        "max_frames",
        "max_images_per_post",
        "max_dynamic_media_per_post",
    }
    _reject_unknown(value, allowed, "encoder")
    representations_value = value.get("representations")
    if not isinstance(representations_value, list) or not representations_value:
        raise FullExperimentConfigurationError(
            "encoder.representations must be a non-empty list"
        )
    representations = tuple(str(item).strip() for item in representations_value)
    if len(set(representations)) != len(representations):
        raise FullExperimentConfigurationError("encoder.representations must be unique")
    for representation in representations:
        if representation != "final" and not _REPRESENTATION_PATTERN.fullmatch(
            representation
        ):
            raise FullExperimentConfigurationError(
                "Representations must be 'final' or an explicit 'layer:N' selector"
            )

    encoder = EncoderConfig(
        model_id=str(value["model_id"]).strip(),
        model_name_or_path=Path(str(value["model_name_or_path"])).expanduser().resolve(),
        revision=str(value["revision"]).strip(),
        representations=representations,
        normalize=bool(value["normalize"]),
        instruction=str(value["instruction"]).strip(),
        torch_dtype=str(value["torch_dtype"]),
        device=str(value["device"]),
        max_length=int(value["max_length"]),
        min_pixels=int(value["min_pixels"]),
        max_pixels=int(value["max_pixels"]),
        total_pixels=int(value["total_pixels"]),
        fps=float(value["fps"]),
        max_frames=int(value["max_frames"]),
        max_images_per_post=int(value["max_images_per_post"]),
        max_dynamic_media_per_post=int(value["max_dynamic_media_per_post"]),
    )
    if not encoder.model_id or not encoder.revision or not encoder.instruction:
        raise FullExperimentConfigurationError(
            "encoder model_id, revision, and instruction must be non-empty"
        )
    if encoder.torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise FullExperimentConfigurationError("Unsupported encoder.torch_dtype")
    if encoder.device not in {"auto", "cpu", "cuda"}:
        raise FullExperimentConfigurationError("encoder.device must be auto, cpu, or cuda")
    positive_fields = (
        encoder.max_length,
        encoder.min_pixels,
        encoder.max_pixels,
        encoder.total_pixels,
        encoder.max_frames,
    )
    if any(item < 1 for item in positive_fields) or encoder.fps <= 0:
        raise FullExperimentConfigurationError("Encoder size and sampling values must be positive")
    if encoder.max_frames < 2:
        raise FullExperimentConfigurationError(
            "encoder.max_frames must be at least 2 for Qwen video inputs"
        )
    if encoder.min_pixels > encoder.max_pixels:
        raise FullExperimentConfigurationError("encoder.min_pixels exceeds max_pixels")
    if encoder.max_images_per_post < 0 or encoder.max_dynamic_media_per_post < 0:
        raise FullExperimentConfigurationError("Per-post media limits cannot be negative")
    return encoder


def _parse_model(payload: Any) -> HierarchicalModelConfig:
    value = _mapping(payload, "model")
    allowed = {
        "d_model",
        "embedding_prefix_dim",
        "relative_position_max_distance",
        "attention_inner_dim",
        "raw_mean_skip_dim",
        "post_attention_heads",
        "user_attention_heads",
        "user_position_encoding",
        "ffn_hidden_dim",
        "classifier_hidden_dim",
        "classifier_head",
        "architecture_variant",
        "dropout",
        "user_pooling",
    }
    _reject_unknown(value, allowed, "model")
    model = HierarchicalModelConfig(
        d_model=int(value["d_model"]),
        post_attention_heads=int(value["post_attention_heads"]),
        user_attention_heads=int(value["user_attention_heads"]),
        user_position_encoding=str(value["user_position_encoding"]),
        ffn_hidden_dim=int(value["ffn_hidden_dim"]),
        classifier_hidden_dim=int(value["classifier_hidden_dim"]),
        dropout=float(value["dropout"]),
        user_pooling=str(value["user_pooling"]),
        embedding_prefix_dim=(
            int(value["embedding_prefix_dim"])
            if value.get("embedding_prefix_dim") is not None
            else None
        ),
        relative_position_max_distance=(
            int(value["relative_position_max_distance"])
            if value.get("relative_position_max_distance") is not None
            else None
        ),
        attention_inner_dim=(
            int(value["attention_inner_dim"])
            if value.get("attention_inner_dim") is not None
            else None
        ),
        raw_mean_skip_dim=(
            int(value["raw_mean_skip_dim"])
            if value.get("raw_mean_skip_dim") is not None
            else None
        ),
        classifier_head=str(value.get("classifier_head", "mlp")),
        architecture_variant=str(value.get("architecture_variant", "full")),
    )
    if min(
        model.d_model,
        model.post_attention_heads,
        model.user_attention_heads,
        model.ffn_hidden_dim,
        model.classifier_hidden_dim,
    ) < 1:
        raise FullExperimentConfigurationError("Model dimensions and counts must be positive")
    if model.attention_inner_dim is not None and model.attention_inner_dim < 1:
        raise FullExperimentConfigurationError(
            "model.attention_inner_dim must be positive when provided"
        )
    if model.raw_mean_skip_dim is not None and model.raw_mean_skip_dim < 1:
        raise FullExperimentConfigurationError(
            "model.raw_mean_skip_dim must be positive when provided"
        )
    attention_inner_dim = model.attention_inner_dim or model.d_model
    if attention_inner_dim % model.post_attention_heads:
        raise FullExperimentConfigurationError(
            "The effective attention inner dimension must be divisible by "
            "post_attention_heads"
        )
    if attention_inner_dim % model.user_attention_heads:
        raise FullExperimentConfigurationError(
            "The effective attention inner dimension must be divisible by "
            "user_attention_heads"
        )
    if model.embedding_prefix_dim is not None and model.embedding_prefix_dim < 1:
        raise FullExperimentConfigurationError(
            "model.embedding_prefix_dim must be positive when provided"
        )
    if model.user_position_encoding not in {"none", "sinusoidal", "relative_bias"}:
        raise FullExperimentConfigurationError(
            "model.user_position_encoding must be none, sinusoidal, or relative_bias"
        )
    if model.user_position_encoding == "relative_bias":
        if (
            model.relative_position_max_distance is None
            or model.relative_position_max_distance < 1
        ):
            raise FullExperimentConfigurationError(
                "relative_bias requires a positive "
                "model.relative_position_max_distance"
            )
    elif model.relative_position_max_distance is not None:
        raise FullExperimentConfigurationError(
            "model.relative_position_max_distance is only valid with relative_bias"
        )
    if not 0 <= model.dropout < 1:
        raise FullExperimentConfigurationError("model.dropout must be in [0, 1)")
    if model.classifier_head not in {"mlp", "linear"}:
        raise FullExperimentConfigurationError(
            "model.classifier_head must be mlp or linear"
        )
    if model.architecture_variant not in {
        "full",
        "without_post_cross_attention",
        "without_user_self_attention",
        "mean_pooling_at_both_levels",
    }:
        raise FullExperimentConfigurationError(
            "model.architecture_variant must be full, without_post_cross_attention, "
            "without_user_self_attention, or mean_pooling_at_both_levels"
        )
    if model.user_pooling != "mean":
        raise FullExperimentConfigurationError(
            "The paper-faithful full experiment currently supports mean user pooling only"
        )
    return model


def _parse_training(payload: Any) -> TrainingConfig:
    value = _mapping(payload, "training")
    allowed = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "early_stopping_patience",
        "selection_metric",
        "seed",
        "num_workers",
    }
    _reject_unknown(value, allowed, "training")
    training = TrainingConfig(
        epochs=int(value["epochs"]),
        batch_size=int(value["batch_size"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        gradient_clip_norm=float(value["gradient_clip_norm"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
        selection_metric=str(value["selection_metric"]),
        seed=int(value["seed"]),
        num_workers=int(value["num_workers"]),
    )
    if min(training.epochs, training.batch_size, training.early_stopping_patience) < 1:
        raise FullExperimentConfigurationError("Training counts must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise FullExperimentConfigurationError("Invalid optimizer values")
    if training.gradient_clip_norm <= 0 or training.num_workers < 0:
        raise FullExperimentConfigurationError("Invalid training runtime values")
    if training.selection_metric != "loss":
        raise FullExperimentConfigurationError(
            "The core full experiment uses validation BCE loss for early stopping"
        )
    return training


def load_full_experiment_config(path: str | Path) -> FullExperimentConfig:
    payload = _mapping(
        _expand_environment(json.loads(Path(path).read_text(encoding="utf-8"))),
        "full experiment configuration",
    )
    _reject_unknown(payload, {"schema_version", "encoder", "model", "training"}, "top-level")
    if payload.get("schema_version") != 2:
        raise FullExperimentConfigurationError(
            "Only full experiment configuration schema_version=2 is supported"
        )
    return FullExperimentConfig(
        schema_version=2,
        encoder=_parse_encoder(payload.get("encoder")),
        model=_parse_model(payload.get("model")),
        training=_parse_training(payload.get("training")),
    )


def parse_hierarchical_model_config(payload: Any) -> HierarchicalModelConfig:
    """Validate a reusable hierarchical-head model section."""
    return _parse_model(payload)


def parse_hierarchical_training_config(payload: Any) -> TrainingConfig:
    """Validate a reusable hierarchical-head training section."""
    return _parse_training(payload)
