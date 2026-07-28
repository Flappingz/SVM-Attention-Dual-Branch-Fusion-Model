from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_raw_group, write_study_config

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.data.prepare import prepare_dataset
from ocd_v3.experiments.full_config import (
    EncoderConfig,
    FullExperimentConfig,
    HierarchicalModelConfig,
    TrainingConfig,
)
from ocd_v3.features.build import build_feature_set
from ocd_v3.features.storage import load_subject_feature_bundle
from ocd_v3.features.training_data import PreparedFeatureSet


class FakeEmbedder:
    hidden_size = 8

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config

    def embed_post(self, value: object) -> object:
        import numpy as np

        text = str(value.text)
        media_count = len(value.images) + len(value.videos)
        vector = np.zeros((1, 8), dtype=np.float32)
        vector[0, 0] = len(text)
        vector[0, 1] = media_count
        return vector

    def manifest(self) -> dict[str, object]:
        return {
            "model_id": "synthetic",
            "revision": "test",
            "representations": ["final"],
            "hidden_size": 8,
            "normalize": True,
            "instruction": "Represent this post.",
        }


def _full_config(root: Path) -> FullExperimentConfig:
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        weights.write_bytes(b"synthetic-weights")
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path.write_text('{"model_type":"synthetic"}', encoding="utf-8")
    return FullExperimentConfig(
        schema_version=2,
        encoder=EncoderConfig(
            model_id="synthetic",
            model_name_or_path=root / "model",
            revision="test",
            representations=("final",),
            normalize=True,
            instruction="Represent this post.",
            torch_dtype="float32",
            device="cpu",
            max_length=128,
            min_pixels=16,
            max_pixels=64,
            total_pixels=64,
            fps=1.0,
            max_frames=2,
            max_images_per_post=1,
            max_dynamic_media_per_post=1,
        ),
        model=HierarchicalModelConfig(
            d_model=4,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            early_stopping_patience=1,
            selection_metric="loss",
            seed=1,
            num_workers=0,
        ),
    )


class FeatureBuildTests(unittest.TestCase):
    def test_reencodes_text_only_and_image_only_without_cross_modal_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)
            common = {
                "dataset": dataset,
                "study_config": study,
                "full_config": _full_config(root),
                "keyword_condition": "original",
                "output_root": artifact_root / "features",
                "embedder_factory": FakeEmbedder,
            }
            text_only = build_feature_set(
                **common, embedding_input_modalities=("text",)
            )
            image_only = build_feature_set(
                **common, embedding_input_modalities=("image",)
            )
            self.assertNotEqual(text_only.feature_dir, image_only.feature_dir)
            self.assertEqual(text_only.manifest["embedding_input_modalities"], ["text"])
            self.assertEqual(image_only.manifest["embedding_input_modalities"], ["image"])
            subject_id = next(dataset.subjects()).subject_id
            text_payload = load_subject_feature_bundle(
                text_only.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            image_payload = load_subject_feature_bundle(
                image_only.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            self.assertEqual(text_payload["modality_presence"][0].tolist(), [1, 0, 0])
            self.assertEqual(image_payload["modality_presence"][0].tolist(), [0, 1, 0])
            self.assertGreater(float(text_payload["content_embeddings"][0, 0, 0]), 0)
            self.assertEqual(float(text_payload["content_embeddings"][0, 0, 1]), 0.0)
            self.assertEqual(float(image_payload["content_embeddings"][0, 0, 0]), 0.0)
            self.assertEqual(float(image_payload["content_embeddings"][0, 0, 1]), 1.0)

    def test_model_snapshot_content_changes_feature_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)
            full_config = _full_config(root)
            first = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="original",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )
            (root / "model" / "model.safetensors").write_bytes(b"changed--weights")
            second = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="original",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )
            self.assertNotEqual(first.feature_dir, second.feature_dir)
            self.assertNotEqual(
                first.manifest["encoder_runtime"]["local_snapshot"]["sha256"],
                second.manifest["encoder_runtime"]["local_snapshot"]["sha256"],
            )

    def test_builds_aligned_original_and_masked_multimodal_bundles(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)
            full_config = _full_config(root)
            original = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="original",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )
            masked = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="masked",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )
            self.assertTrue(original.manifest["complete_dataset"])
            self.assertEqual(original.manifest["counts"]["selected_images"], 2)
            self.assertEqual(original.manifest["counts"]["dropped_images_by_cap"], 0)
            self.assertNotEqual(original.feature_dir, masked.feature_dir)
            subject_id = next(dataset.subjects()).subject_id
            original_payload = load_subject_feature_bundle(
                original.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            masked_payload = load_subject_feature_bundle(
                masked.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            np.testing.assert_array_equal(
                original_payload["post_ids"], masked_payload["post_ids"]
            )
            self.assertEqual(original_payload["modality_presence"][0].tolist(), [1, 1, 0])
            self.assertNotEqual(
                original_payload["content_embeddings"][0, 0, 0],
                masked_payload["content_embeddings"][0, 0, 0],
            )

    def test_interrupted_build_reuses_completed_subject_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)
            full_config = _full_config(root)

            class InterruptingEmbedder(FakeEmbedder):
                calls = 0

                def embed_post(self, value: object) -> object:
                    type(self).calls += 1
                    if type(self).calls == 2:
                        raise RuntimeError("synthetic interruption")
                    return super().embed_post(value)

            with self.assertRaises(RuntimeError):
                build_feature_set(
                    dataset=dataset,
                    study_config=study,
                    full_config=full_config,
                    keyword_condition="original",
                    output_root=artifact_root / "features",
                    embedder_factory=InterruptingEmbedder,
                )

            class CountingEmbedder(FakeEmbedder):
                calls = 0

                def embed_post(self, value: object) -> object:
                    type(self).calls += 1
                    return super().embed_post(value)

            resumed = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="original",
                output_root=artifact_root / "features",
                embedder_factory=CountingEmbedder,
            )
            self.assertTrue(resumed.manifest["complete_dataset"])
            self.assertEqual(CountingEmbedder.calls, 1)

    def test_removed_condition_retains_a_subject_with_no_remaining_posts(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)

            removed = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=_full_config(root),
                keyword_condition="removed",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )

            self.assertTrue(removed.manifest["keyword_post_removal"]["active"])
            self.assertEqual(removed.manifest["counts"]["subjects"], 2)
            self.assertEqual(removed.manifest["counts"]["posts"], 0)
            self.assertEqual(
                removed.manifest["counts"]["subjects_with_no_posts_in_removed_model_window"],
                2,
            )
            subject_id = next(dataset.subjects()).subject_id
            payload = load_subject_feature_bundle(
                removed.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            self.assertEqual(payload["post_ids"].shape, (0,))
            self.assertEqual(payload["metadata"].shape, (0, 5))
            self.assertEqual(payload["content_embeddings"].shape, (0, 1, 8))
            self.assertEqual(payload["modality_presence"].shape, (0, 3))
            self.assertTrue(np.isfinite(payload["content_embeddings"]).all())

    def test_removed_condition_reuses_original_window_embeddings(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            artifact_root = root / "artifacts"
            make_raw_group(raw_root, "control", ["100"])
            make_raw_group(raw_root, "self_reporting_ocd", ["200"])
            study_path = root / "study.json"
            write_study_config(study_path, raw_root, artifact_root)
            study = load_config(study_path)
            dataset_dir, _ = prepare_dataset(study)
            dataset = PreparedDataset(dataset_dir)
            full_config = _full_config(root)
            original = build_feature_set(
                dataset=dataset,
                study_config=study,
                full_config=full_config,
                keyword_condition="original",
                output_root=artifact_root / "features",
                embedder_factory=FakeEmbedder,
            )

            replacement_study_payload = json.loads(study_path.read_text(encoding="utf-8"))
            replacement_study_payload["dataset"]["keywords"] = ["not-present"]
            replacement_study_path = root / "removed-study.json"
            replacement_study_path.write_text(
                json.dumps(replacement_study_payload), encoding="utf-8"
            )
            replacement_study = load_config(replacement_study_path)

            class FailingEmbedder:
                def __init__(self, config: object) -> None:
                    raise AssertionError("Embedding should have been reused")

            removed = build_feature_set(
                dataset=dataset,
                study_config=replacement_study,
                full_config=full_config,
                keyword_condition="removed",
                output_root=artifact_root / "features",
                source_feature_set=PreparedFeatureSet(original.feature_dir),
                embedder_factory=FailingEmbedder,
            )

            self.assertEqual(removed.manifest["build_runtime"]["embedded_posts_this_invocation"], 0)
            self.assertEqual(removed.manifest["counts"]["reused_source_feature_posts"], 2)
            subject_id = next(dataset.subjects()).subject_id
            original_payload = load_subject_feature_bundle(
                original.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            removed_payload = load_subject_feature_bundle(
                removed.feature_dir / "subjects" / f"{subject_id}.npz"
            )
            np.testing.assert_array_equal(
                original_payload["content_embeddings"], removed_payload["content_embeddings"]
            )


if __name__ == "__main__":
    unittest.main()
