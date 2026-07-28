"""Fixed-fold Qwen3-VL final-embedding mean-pooling MLP baseline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ocd_v3.config import StudyConfig
from ocd_v3.experiments.qwen_vl_mean_mlp_config import QwenVLMeanMLPConfig
from ocd_v3.experiments.train_full import TrainFullResult, train_hierarchical_full
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.modeling.qwen_vl_mean_mlp import QwenVLMeanMLPClassifier

QWEN_VL_MEAN_MLP_EXPERIMENT_ID = "qwen3_vl_final_embedding_mean_pooling_mlp"


def _validate_feature_encoder(
    feature_set: PreparedFeatureSet, configuration: QwenVLMeanMLPConfig
) -> dict[str, Any]:
    if feature_set.schema.representation_names != ("final",):
        raise ValueError("Qwen3-VL mean-MLP features require exactly the final representation")
    encoder = feature_set.manifest.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("Feature manifest has no encoder record")
    for field, expected in (
        ("model_id", configuration.encoder.model_id),
        ("revision", configuration.encoder.revision),
        ("normalize", configuration.encoder.normalize),
    ):
        if encoder.get(field) != expected:
            raise ValueError(f"Feature encoder {field} differs from the baseline configuration")
    return encoder


def train_qwen_vl_mean_mlp(
    *,
    feature_set: PreparedFeatureSet,
    split_assignments_path: Path,
    study_config: StudyConfig,
    configuration: QwenVLMeanMLPConfig,
    output_root: Path,
    selected_folds: Iterable[int] | None = None,
    epoch_limit: int | None = None,
    run_context: dict[str, Any] | None = None,
) -> TrainFullResult:
    encoder_manifest = _validate_feature_encoder(feature_set, configuration)
    additional_parameters: dict[str, Any] = {
        "source_encoder": encoder_manifest,
        "source_modalities": ["text", "image", "live_photo"],
        "representation_pooling": "Qwen last valid token final embedding",
        "user_pooling": "unweighted masked mean over selected posts",
    }
    if run_context is not None:
        additional_parameters["run_context"] = run_context
    return train_hierarchical_full(
        feature_set=feature_set,
        split_assignments_path=split_assignments_path,
        study_config=study_config,
        full_config=configuration,
        output_root=output_root,
        selected_folds=selected_folds,
        epoch_limit=epoch_limit,
        experiment_id=QWEN_VL_MEAN_MLP_EXPERIMENT_ID,
        configuration_parameter_name="qwen_vl_mean_mlp_baseline",
        additional_parameters=additional_parameters,
        summary_fields={
            "experiment_id": QWEN_VL_MEAN_MLP_EXPERIMENT_ID,
            "architecture": "final_embedding_masked_mean_mlp",
            "metadata_condition": "excluded",
        },
        model_factory=lambda: QwenVLMeanMLPClassifier(
            embedding_dimension=feature_set.schema.embedding_dimension,
            config=configuration.model,
        ),
        include_metadata=False,
    )
