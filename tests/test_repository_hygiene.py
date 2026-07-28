from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ocd_v3.experiments.full_config import (
    EncoderConfig,
    FullExperimentConfig,
    HierarchicalModelConfig,
    TrainingConfig,
)
from ocd_v3.features.qwen_vl import Qwen3VLEmbedder

REPOSITORY = Path(__file__).resolve().parents[1]


def _encoder(model_path: Path) -> EncoderConfig:
    return EncoderConfig(
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        model_name_or_path=model_path,
        revision="test-revision",
        representations=("final",),
        normalize=True,
        instruction="Represent the post for retrieval.",
        torch_dtype="float32",
        device="cpu",
        max_length=32,
        min_pixels=1,
        max_pixels=1,
        total_pixels=1,
        fps=1.0,
        max_frames=2,
        max_images_per_post=1,
        max_dynamic_media_per_post=1,
    )


class RepositoryHygieneTests(unittest.TestCase):
    def test_manifest_config_omits_local_model_path(self) -> None:
        config = FullExperimentConfig(
            schema_version=1,
            encoder=_encoder(Path("/private/local/model")),
            model=HierarchicalModelConfig(
                d_model=16,
                post_attention_heads=1,
                user_attention_heads=1,
                user_position_encoding="none",
                ffn_hidden_dim=32,
                classifier_hidden_dim=16,
                dropout=0.1,
                user_pooling="mean",
            ),
            training=TrainingConfig(
                epochs=1,
                batch_size=1,
                learning_rate=1e-4,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                early_stopping_patience=1,
                selection_metric="f1",
                seed=42,
                num_workers=0,
            ),
        )

        payload = config.public_dict()

        self.assertNotIn("model_name_or_path", payload["encoder"])
        self.assertNotIn("/private/local/model", json.dumps(payload))

    def test_qwen_manifest_records_snapshot_size_without_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.safetensors").write_bytes(b"abc")
            embedder = object.__new__(Qwen3VLEmbedder)
            embedder.config = _encoder(model_path)
            embedder.hidden_size = 8
            embedder.transformer_layer_count = 2
            embedder.attention_implementation = None
            embedder.device = SimpleNamespace(type="cpu")

            with patch(
                "ocd_v3.features.qwen_vl.importlib.metadata.version",
                return_value="test-version",
            ):
                manifest = embedder.manifest()

        self.assertEqual(manifest["local_snapshot"]["file_count"], 1)
        self.assertEqual(manifest["local_snapshot"]["total_bytes"], 3)
        self.assertEqual(len(manifest["local_snapshot"]["sha256"]), 64)
        self.assertNotIn(str(model_path), json.dumps(manifest))
        self.assertNotIn("config_path", manifest["local_snapshot"])

    def test_fusion_audit_has_no_private_feature_id_constant(self) -> None:
        source = (
            REPOSITORY / "tools" / "audit_qwen_sequence_fusion_confirmation.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("EXPECTED_FEATURE_ID", source)
        self.assertIn("--expected-feature-id", source)
        self.assertIn("expected_feature_id: str | None = None", source)

    def test_ci_runs_repository_checks(self) -> None:
        workflow = (REPOSITORY / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("python tools/repository_check.py", workflow)


if __name__ == "__main__":
    unittest.main()
