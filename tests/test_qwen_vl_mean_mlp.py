from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import write_study_config

from ocd_v3.config import load_config
from ocd_v3.data.dataset import PreparedDataset
from ocd_v3.experiments.full_config import TrainingConfig
from ocd_v3.experiments.qwen_vl_mean_mlp import train_qwen_vl_mean_mlp
from ocd_v3.experiments.qwen_vl_mean_mlp_config import (
    QwenVLMeanMLPConfig,
    QwenVLMeanMLPEncoderConfig,
    QwenVLMeanMLPModelConfig,
    load_qwen_vl_mean_mlp_config,
)
from ocd_v3.experiments.qwen_vl_mean_mlp_repeated_cv import (
    run_repeated_qwen_vl_mean_mlp_cv,
)
from ocd_v3.features.storage import save_subject_feature_bundle
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import write_jsonl_atomic
from ocd_v3.modeling.qwen_vl_mean_mlp import QwenVLMeanMLPClassifier


def _configuration() -> QwenVLMeanMLPConfig:
    return QwenVLMeanMLPConfig(
        schema_version=1,
        encoder=QwenVLMeanMLPEncoderConfig(
            model_id="Qwen/test-embedding",
            revision="fixed-revision",
            representations=("final",),
            normalize=True,
            device="cpu",
            frozen=True,
            metadata="excluded",
        ),
        model=QwenVLMeanMLPModelConfig(
            post_pooling="masked_mean",
            input_normalization="layer_norm",
            hidden_dimension=4,
            activation="gelu",
            dropout=0.0,
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


def _write_features(root: Path) -> Path:
    import numpy as np

    feature_dir = root / "features" / "features-qwen-test"
    for index in range(10):
        subject_id = f"subject-{index:02d}"
        label = index % 2
        embeddings = np.asarray(
            [
                [[float(label), index / 10, 0.0, 1.0]],
                [[float(label), index / 10, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        save_subject_feature_bundle(
            feature_dir / "subjects" / f"{subject_id}.npz",
            subject_id=subject_id,
            post_ids=[f"post-{index}-0", f"post-{index}-1"],
            metadata=[[index, 0, 0, 0, 1], [index + 1, 0, 0, 1, 0]],
            content_embeddings=embeddings,
            representation_names=["final"],
            modality_presence=[[True, False, False], [True, False, False]],
            modality_names=["text", "image", "live_photo"],
            encoder_manifest={
                "model_id": "Qwen/test-embedding",
                "revision": "fixed-revision",
                "normalize": True,
            },
        )
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "feature_id": "features-qwen-test",
                "dataset_id": "dataset-qwen-test",
                "keyword_condition": "original",
                "complete_dataset": True,
                "post_selection": {
                    "maximum_posts_per_subject": 2,
                    "strategy": "most_recent",
                },
                "encoder": {
                    "model_id": "Qwen/test-embedding",
                    "revision": "fixed-revision",
                    "normalize": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return feature_dir


def _write_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "dataset-qwen-test",
                "post_selection_for_models": {
                    "maximum_posts_per_subject": 2,
                    "strategy": "most_recent",
                },
            }
        ),
        encoding="utf-8",
    )
    write_jsonl_atomic(
        dataset_dir / "subjects.jsonl",
        [
            {
                "subject_id": f"subject-{index:02d}",
                "label_name": "self_reported_ocd" if index % 2 else "control",
                "label_id": index % 2,
                "post_count": 2,
                "media_count": 0,
                "eligible": True,
            }
            for index in range(10)
        ],
    )
    return dataset_dir


def _write_split(root: Path) -> Path:
    subjects = [f"subject-{index:02d}" for index in range(10)]
    assignments: list[dict[str, object]] = []
    for fold in range(5):
        test_ids = {subjects[2 * fold], subjects[2 * fold + 1]}
        remaining = [subject for subject in subjects if subject not in test_ids]
        validation_ids = {
            next(subject for subject in remaining if int(subject[-2:]) % 2 == 0),
            next(subject for subject in remaining if int(subject[-2:]) % 2 == 1),
        }
        for subject_id in subjects:
            label = int(subject_id[-2:]) % 2
            assignments.append(
                {
                    "outer_fold": fold,
                    "subject_id": subject_id,
                    "label_name": "self_reported_ocd" if label else "control",
                    "label_id": label,
                    "role": (
                        "test"
                        if subject_id in test_ids
                        else "validation"
                        if subject_id in validation_ids
                        else "train"
                    ),
                    "available_post_count": 2,
                    "selected_post_count": 2,
                }
            )
    path = root / "splits" / "assignments.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_id": "splits-qwen-test",
                "dataset_id": "dataset-qwen-test",
                "seed": 99,
                "assignments": assignments,
            }
        ),
        encoding="utf-8",
    )
    return path


class QwenVLMeanMLPTests(unittest.TestCase):
    def test_masked_mean_ignores_padding_and_metadata_is_rejected(self) -> None:
        import torch

        model = QwenVLMeanMLPClassifier(
            embedding_dimension=2,
            config=QwenVLMeanMLPModelConfig(
                post_pooling="masked_mean",
                input_normalization="layer_norm",
                hidden_dimension=2,
                activation="gelu",
                dropout=0.0,
            ),
        )
        content = torch.tensor([[[[2.0, 4.0]], [[4.0, 8.0]], [[100.0, 100.0]]]])
        mask = torch.tensor([[True, True, False]])
        result = model(content, torch.empty((1, 3, 0)), mask)
        self.assertEqual(result["user_embedding"].tolist(), [[3.0, 6.0]])
        with self.assertRaisesRegex(ValueError, "excludes metadata"):
            model(content, torch.zeros((1, 3, 1)), mask)

    def test_five_fold_training_writes_oof_without_metadata_scalers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = train_qwen_vl_mean_mlp(
                feature_set=PreparedFeatureSet(_write_features(root)),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                configuration=_configuration(),
                output_root=root / "runs",
                run_context={"controlled_benchmark": True},
            )
            self.assertTrue(result.summary["complete_outer_cv"])
            self.assertEqual(result.summary["oof_metrics"]["n"], 10)
            self.assertEqual(result.summary["metadata_condition"], "excluded")
            self.assertFalse(
                (result.run_dir / "fold-0" / "metadata_standardizer.json").exists()
            )
            manifest = json.loads(
                (result.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["experiment_id"],
                "qwen3_vl_final_embedding_mean_pooling_mlp",
            )
            self.assertEqual(
                manifest["parameters"]["run_context"],
                {"controlled_benchmark": True},
            )

            repeated = run_repeated_qwen_vl_mean_mlp_cv(
                dataset=PreparedDataset(_write_dataset(root)),
                feature_set=PreparedFeatureSet(root / "features" / "features-qwen-test"),
                study_config=load_config(study_path),
                configuration=_configuration(),
                run_output_root=root / "runs",
                split_output_root=root / "repeated-splits",
                repeated_cv_output_root=root / "repeated-cv",
                split_seeds=(42, 43),
            )
            self.assertEqual(repeated.summary["n_split_repetitions"], 2)
            self.assertEqual(repeated.summary["trained_fold_models"], 10)
            self.assertEqual(repeated.summary["oof_subjects_per_repetition"], 10)

    def test_strict_configuration_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.json"
            path.write_text(json.dumps(_configuration().public_dict()), encoding="utf-8")
            loaded = load_qwen_vl_mean_mlp_config(path)
            self.assertEqual(loaded.encoder.representations, ("final",))
            self.assertEqual(loaded.model.hidden_dimension, 4)


if __name__ == "__main__":
    unittest.main()
