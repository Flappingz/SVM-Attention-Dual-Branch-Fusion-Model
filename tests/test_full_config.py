from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocd_v3.experiments.full_config import (
    FullExperimentConfigurationError,
    load_full_experiment_config,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "encoder": {
            "model_id": "synthetic/qwen",
            "model_name_or_path": "${SYNTHETIC_MODEL}",
            "revision": "revision-test",
            "representations": ["final"],
            "normalize": True,
            "instruction": "Represent this post.",
            "torch_dtype": "float32",
            "device": "cpu",
            "max_length": 128,
            "min_pixels": 16,
            "max_pixels": 64,
            "total_pixels": 64,
            "fps": 1.0,
            "max_frames": 2,
            "max_images_per_post": 2,
            "max_dynamic_media_per_post": 1,
        },
        "model": {
            "d_model": 4,
            "post_attention_heads": 2,
            "user_attention_heads": 2,
            "user_position_encoding": "sinusoidal",
            "ffn_hidden_dim": 8,
            "classifier_hidden_dim": 8,
            "dropout": 0.0,
            "user_pooling": "mean",
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "selection_metric": "loss",
            "seed": 3,
            "num_workers": 0,
        },
    }


class FullConfigTests(unittest.TestCase):
    def test_loads_paper_faithful_config_with_local_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "full.json"
            config_path.write_text(json.dumps(_payload()), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.encoder.representations, ("final",))
            self.assertEqual(config.model.d_model, 4)
            self.assertIsNone(config.model.embedding_prefix_dim)
            self.assertIsNone(config.model.relative_position_max_distance)
            self.assertIsNone(config.model.attention_inner_dim)
            self.assertIsNone(config.model.raw_mean_skip_dim)
            self.assertEqual(config.model.classifier_head, "mlp")
            self.assertEqual(config.model.architecture_variant, "full")
            self.assertNotIn(
                "embedding_prefix_dim", config.public_dict()["model"]
            )
            self.assertNotIn(
                "relative_position_max_distance", config.public_dict()["model"]
            )
            self.assertNotIn("attention_inner_dim", config.public_dict()["model"])
            self.assertNotIn("raw_mean_skip_dim", config.public_dict()["model"])
            self.assertNotIn("classifier_head", config.public_dict()["model"])
            self.assertNotIn("architecture_variant", config.public_dict()["model"])
            self.assertEqual(config.model.user_position_encoding, "sinusoidal")
            self.assertEqual(config.encoder.model_name_or_path, (root / "model").resolve())
            self.assertNotIn("model_name_or_path", config.public_dict()["encoder"])

    def test_rejects_ambiguous_layer_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["encoder"]["representations"] = ["last"]  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_loads_explicit_embedding_prefix_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["embedding_prefix_dim"] = 6  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.embedding_prefix_dim, 6)
            self.assertEqual(config.model.d_model, 4)
            self.assertEqual(config.public_dict()["model"]["embedding_prefix_dim"], 6)

    def test_rejects_nonpositive_embedding_prefix_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["embedding_prefix_dim"] = 0  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_loads_linear_classifier_head_without_changing_other_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["classifier_head"] = "linear"  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.classifier_head, "linear")
            self.assertEqual(config.model.d_model, 4)
            self.assertEqual(config.model.classifier_hidden_dim, 8)
            self.assertEqual(config.public_dict()["model"]["classifier_head"], "linear")

    def test_loads_compact_attention_inner_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["attention_inner_dim"] = 2  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.d_model, 4)
            self.assertEqual(config.model.attention_inner_dim, 2)
            self.assertEqual(config.public_dict()["model"]["attention_inner_dim"], 2)

    def test_loads_raw_qwen_mean_skip_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["raw_mean_skip_dim"] = 3  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.raw_mean_skip_dim, 3)
            self.assertEqual(config.public_dict()["model"]["raw_mean_skip_dim"], 3)

    def test_rejects_attention_inner_dimension_not_divisible_by_heads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["attention_inner_dim"] = 3  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_rejects_unknown_classifier_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["classifier_head"] = "deep"  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_loads_and_validates_architecture_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["architecture_variant"] = (  # type: ignore[index]
                "without_post_cross_attention"
            )
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(
                config.model.architecture_variant, "without_post_cross_attention"
            )
            self.assertEqual(
                config.public_dict()["model"]["architecture_variant"],
                "without_post_cross_attention",
            )

            payload["model"]["architecture_variant"] = "unknown"  # type: ignore[index]
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_loads_relative_position_bias_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["user_position_encoding"] = "relative_bias"  # type: ignore[index]
            payload["model"]["relative_position_max_distance"] = 63  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.user_position_encoding, "relative_bias")
            self.assertEqual(config.model.relative_position_max_distance, 63)
            self.assertEqual(
                config.public_dict()["model"]["relative_position_max_distance"],
                63,
            )

    def test_loads_explicit_no_position_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["user_position_encoding"] = "none"  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                config = load_full_experiment_config(config_path)
            self.assertEqual(config.model.user_position_encoding, "none")
            self.assertIsNone(config.model.relative_position_max_distance)

    def test_relative_position_bias_requires_maximum_distance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["user_position_encoding"] = "relative_bias"  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)

    def test_rejects_removed_latent_query_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _payload()
            payload["model"]["post_query_count"] = 4  # type: ignore[index]
            config_path = root / "full.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(os.environ, {"SYNTHETIC_MODEL": str(root / "model")}):
                with self.assertRaises(FullExperimentConfigurationError):
                    load_full_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
