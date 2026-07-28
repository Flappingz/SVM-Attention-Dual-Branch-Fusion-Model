from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import write_study_config

from ocd_v3.config import load_config
from ocd_v3.experiments.full_config import (
    EncoderConfig,
    FullExperimentConfig,
    HierarchicalModelConfig,
    TrainingConfig,
)
from ocd_v3.experiments.train_full import train_hierarchical_full
from ocd_v3.features.storage import save_subject_feature_bundle
from ocd_v3.features.training_data import PreparedFeatureSet


def _full_config(root: Path) -> FullExperimentConfig:
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
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            early_stopping_patience=1,
            selection_metric="loss",
            seed=11,
            num_workers=0,
        ),
    )


class TrainFullTests(unittest.TestCase):
    def test_five_fold_training_writes_complete_oof_artifacts(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            study = load_config(study_path)
            feature_dir = root / "features" / "features-test"
            bundle_dir = feature_dir / "subjects"
            subjects = [f"sub-{index:02d}" for index in range(10)]
            labels = {subject_id: index % 2 for index, subject_id in enumerate(subjects)}
            for index, subject_id in enumerate(subjects):
                embedding = np.full((2, 1, 8), labels[subject_id], dtype=np.float32)
                embedding[:, :, 1] = index / 10
                save_subject_feature_bundle(
                    bundle_dir / f"{subject_id}.npz",
                    subject_id=subject_id,
                    post_ids=[f"post-{index}-0", f"post-{index}-1"],
                    metadata=[
                        [index, 0, 0, 0, 1],
                        [index + 1, 0, 0, 1, 0],
                    ],
                    content_embeddings=embedding,
                    representation_names=["final"],
                    modality_presence=[[True, False, False], [True, False, False]],
                    modality_names=["text", "image", "live_photo"],
                    encoder_manifest={"model_id": "synthetic", "revision": "test"},
                )
            manifest = {
                "schema_version": 1,
                "feature_id": "features-test",
                "dataset_id": "dataset-test",
                "keyword_condition": "original",
                "complete_dataset": True,
                "post_selection": {"maximum_posts_per_subject": 2, "strategy": "most_recent"},
            }
            feature_dir.mkdir(parents=True, exist_ok=True)
            (feature_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            assignments: list[dict[str, object]] = []
            for fold in range(5):
                test_ids = {subjects[2 * fold], subjects[2 * fold + 1]}
                remaining = [subject for subject in subjects if subject not in test_ids]
                validation_ids = {
                    next(subject for subject in remaining if labels[subject] == 0),
                    next(subject for subject in remaining if labels[subject] == 1),
                }
                for subject_id in subjects:
                    role = (
                        "test"
                        if subject_id in test_ids
                        else "validation"
                        if subject_id in validation_ids
                        else "train"
                    )
                    assignments.append(
                        {
                            "outer_fold": fold,
                            "subject_id": subject_id,
                            "label_name": "self_reported_ocd" if labels[subject_id] else "control",
                            "label_id": labels[subject_id],
                            "role": role,
                            "available_post_count": 2,
                            "selected_post_count": 2,
                        }
                    )
            split_path = root / "splits" / "assignments.json"
            split_path.parent.mkdir(parents=True)
            split_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split_id": "splits-test",
                        "dataset_id": "dataset-test",
                        "seed": 99,
                        "assignments": assignments,
                    }
                ),
                encoding="utf-8",
            )
            result = train_hierarchical_full(
                feature_set=PreparedFeatureSet(feature_dir),
                split_assignments_path=split_path,
                study_config=study,
                full_config=_full_config(root),
                output_root=root / "runs",
            )
            self.assertTrue(result.summary["complete_outer_cv"])
            self.assertEqual(result.summary["oof_metrics"]["n"], 10)
            self.assertEqual(len(result.summary["folds"]), 5)
            self.assertEqual(
                result.summary["trainable_parameter_counts"]["total"],
                result.summary["trainable_parameter_count"],
            )
            self.assertGreater(
                result.summary["trainable_parameter_counts"]["classifier_head"], 0
            )
            self.assertTrue((result.run_dir / "oof_predictions.jsonl").is_file())
            self.assertTrue((result.run_dir / "fold-0" / "best_model.safetensors").is_file())
            run_manifest = json.loads(
                (result.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["seeds"]["split"], 99)


if __name__ == "__main__":
    unittest.main()
