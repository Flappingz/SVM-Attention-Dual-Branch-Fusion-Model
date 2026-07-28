from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import write_study_config

from ocd_v3.config import load_config
from ocd_v3.experiments.qwen_sequence_svm import (
    QWEN_SEQUENCE_PREFIX_SVM_EXPERIMENT_ID,
    qwen_sequence_svm_experiment_id,
    train_qwen_sequence_svm,
)
from ocd_v3.experiments.qwen_sequence_svm_config import (
    QwenSequenceEncoderConfig,
    QwenSequencePoolingConfig,
    QwenSequenceSVMClassifierConfig,
    QwenSequenceSVMConfig,
    load_qwen_sequence_svm_config,
)
from ocd_v3.experiments.qwen_sequence_svm_repeated_cv import (
    _audit_feature_sequences,
)
from ocd_v3.features.qwen_sequence import (
    qwen_mean_std_temporal_halves,
    qwen_segment_mean_std2,
    qwen_temporal_pyramid2,
)
from ocd_v3.features.storage import save_subject_feature_bundle
from ocd_v3.features.training_data import PreparedFeatureSet


def _configuration() -> QwenSequenceSVMConfig:
    return QwenSequenceSVMConfig(
        schema_version=1,
        encoder=QwenSequenceEncoderConfig(
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            revision="synthetic-fixed-revision",
            representation="final",
            embedding_dimension=2048,
            normalize=True,
            frozen=True,
        ),
        sequence_pooling=QwenSequencePoolingConfig(
            method="mean_std_temporal_halves",
            chronological_blocks=(
                "global_mean",
                "global_std",
                "early_half_mean",
                "late_half_mean",
            ),
            block_normalization="l2",
            final_normalization="l2",
        ),
        classifier=QwenSequenceSVMClassifierConfig(
            estimator="linear_svc",
            c=0.1,
            loss="squared_hinge",
            class_weight=None,
            tolerance=1e-4,
            maximum_iterations=10000,
            random_state=42,
        ),
    )


def _write_features(root: Path, *, keyword_condition: str = "removed") -> Path:
    import numpy as np

    feature_dir = root / "features"
    for index in range(10):
        subject_id = f"subject-{index:02d}"
        label = index % 2
        embeddings = np.zeros((4, 1, 2048), dtype=np.float32)
        embeddings[:, 0, 0] = 1.0 if label else -1.0
        embeddings[:2, 0, 1] = -0.5
        embeddings[2:, 0, 1] = 0.5
        save_subject_feature_bundle(
            feature_dir / "subjects" / f"{subject_id}.npz",
            subject_id=subject_id,
            post_ids=[f"post-{index}-{post}" for post in range(4)],
            metadata=[[0, 0, 0, 0, 1]] * 4,
            content_embeddings=embeddings,
            representation_names=["final"],
            modality_presence=[[True, False, False]] * 4,
            modality_names=["text", "image", "live_photo"],
            encoder_manifest={
                "model_id": "Qwen/Qwen3-VL-Embedding-2B",
                "revision": "synthetic-fixed-revision",
                "normalize": True,
            },
        )
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "feature_id": "features-sequence-test",
                "dataset_id": "dataset-sequence-test",
                "keyword_condition": keyword_condition,
                "keyword_policy": {"keywords": ["ocd"], "replacement": "[MASK]"},
                "keyword_post_removal": {
                    "active": keyword_condition == "removed",
                    "rule": "synthetic fixed-window complete-post removal",
                    "users_without_posts_are_retained": True,
                },
                "counts": {
                    "keyword_matched_posts_removed_from_model_window": (
                        1 if keyword_condition == "removed" else 0
                    ),
                },
                "complete_dataset": True,
                "post_selection": {
                    "maximum_posts_per_subject": 64,
                    "strategy": "most_recent",
                },
                "encoder": {
                    "model_id": "Qwen/Qwen3-VL-Embedding-2B",
                    "revision": "synthetic-fixed-revision",
                    "normalize": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return feature_dir


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
                    "available_post_count": 4,
                    "selected_post_count": 4,
                }
            )
    path = root / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_id": "splits-sequence-test",
                "dataset_id": "dataset-sequence-test",
                "seed": 99,
                "assignments": assignments,
            }
        ),
        encoding="utf-8",
    )
    return path


class QwenSequenceSVMTests(unittest.TestCase):
    def test_repeated_input_audit_uses_feature_keyword_condition(self) -> None:
        class SyntheticDataset:
            def __init__(self) -> None:
                self.conditions: list[str] = []

            def selected_posts(
                self,
                subject_id: str,
                *,
                keyword_condition: str,
                keywords: tuple[str, ...],
                replacement: str,
            ) -> list[dict[str, str]]:
                self.conditions.append(keyword_condition)
                index = int(subject_id[-2:])
                return [
                    {
                        "post_id": f"post-{index}-{post}",
                        "cleaned_text": "safe synthetic text",
                    }
                    for post in range(4)
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            dataset = SyntheticDataset()
            feature_set = PreparedFeatureSet(
                _write_features(root, keyword_condition="original")
            )
            subject_ids = [f"subject-{index:02d}" for index in range(10)]
            _audit_feature_sequences(
                dataset=dataset,  # type: ignore[arg-type]
                feature_set=feature_set,
                study_config=load_config(study_path),
                subject_ids=subject_ids,
            )
            self.assertEqual(dataset.conditions, ["original"] * 10)

    def test_sequence_summary_uses_temporal_order_and_stays_finite(self) -> None:
        import numpy as np

        sequence = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 4.0]])
        original = qwen_mean_std_temporal_halves(
            sequence, expected_embedding_dimension=2
        )
        reversed_sequence = qwen_mean_std_temporal_halves(
            sequence[::-1], expected_embedding_dimension=2
        )
        self.assertEqual(original.shape, (8,))
        self.assertTrue(np.isfinite(original).all())
        self.assertFalse(np.array_equal(original, reversed_sequence))
        pyramid = qwen_temporal_pyramid2(
            sequence, expected_embedding_dimension=2
        )
        reversed_pyramid = qwen_temporal_pyramid2(
            sequence[::-1], expected_embedding_dimension=2
        )
        self.assertEqual(pyramid.shape, (6,))
        self.assertFalse(np.array_equal(pyramid, reversed_pyramid))
        segment_moments = qwen_segment_mean_std2(
            sequence, expected_embedding_dimension=2
        )
        reversed_segment_moments = qwen_segment_mean_std2(
            sequence[::-1], expected_embedding_dimension=2
        )
        self.assertEqual(segment_moments.shape, (12,))
        self.assertTrue(np.isfinite(segment_moments).all())
        self.assertFalse(np.array_equal(segment_moments, reversed_segment_moments))

    def test_segment_moment_config_supports_an_explicit_embedding_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text-segment.json"
            payload = _configuration().public_dict()
            payload["sequence_pooling"] = {
                "method": "segment_mean_std2",
                "chronological_blocks": [
                    "global_mean",
                    "global_std",
                    "early_half_mean",
                    "late_half_mean",
                    "early_half_std",
                    "late_half_std",
                ],
                "block_normalization": "l2",
                "final_normalization": "l2",
                "embedding_prefix_dimension": 256,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            configuration = load_qwen_sequence_svm_config(path)
            self.assertEqual(
                configuration.sequence_pooling.embedding_prefix_dimension, 256
            )
            self.assertEqual(
                qwen_sequence_svm_experiment_id(configuration),
                QWEN_SEQUENCE_PREFIX_SVM_EXPERIMENT_ID,
            )

    def test_five_fold_original_qwen_sequence_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = train_qwen_sequence_svm(
                feature_set=PreparedFeatureSet(
                    _write_features(root, keyword_condition="original")
                ),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                configuration=_configuration(),
                output_root=root / "runs",
                selection_provenance="LOCKED_ORIGINAL_SYNTHETIC_TEST",
            )
            self.assertTrue(result.summary["complete_outer_cv"])
            audit = json.loads(
                (result.run_dir / "sequence_input_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["keyword_condition"], "original")
            self.assertEqual(
                audit["keyword_matched_posts_removed_from_model_window"], 0
            )

    def test_strict_config_and_five_fold_removed_qwen_sequence_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "sequence.json"
            config_path.write_text(
                json.dumps(_configuration().public_dict()), encoding="utf-8"
            )
            configuration = load_qwen_sequence_svm_config(config_path)
            self.assertEqual(configuration.encoder.embedding_dimension, 2048)

            study_path = root / "study.json"
            write_study_config(study_path, root / "raw", root / "artifacts")
            result = train_qwen_sequence_svm(
                feature_set=PreparedFeatureSet(_write_features(root)),
                split_assignments_path=_write_split(root),
                study_config=load_config(study_path),
                configuration=configuration,
                output_root=root / "runs",
                selection_provenance="EXPLORATORY_SYNTHETIC_TEST",
            )
            self.assertTrue(result.summary["complete_outer_cv"])
            self.assertEqual(result.summary["oof_metrics"]["n"], 10)
            self.assertEqual(result.summary["sequence_feature_dimension"], 8192)
            self.assertEqual(result.summary["model_parameter_count"], 8193)
            audit = json.loads(
                (result.run_dir / "sequence_input_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["keyword_condition"], "removed")
            self.assertTrue(audit["chronological_sequence"])
            self.assertEqual(audit["embedding_prefix_dimension"], 2048)
            manifest = json.loads(
                (result.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["parameters"]["model_selection_provenance"],
                "EXPLORATORY_SYNTHETIC_TEST",
            )
            import numpy as np

            with np.load(
                result.run_dir / "fold-0" / "model_parameters.npz",
                allow_pickle=False,
            ) as checkpoint:
                self.assertEqual(checkpoint["coefficient"].shape, (1, 8192))


if __name__ == "__main__":
    unittest.main()
