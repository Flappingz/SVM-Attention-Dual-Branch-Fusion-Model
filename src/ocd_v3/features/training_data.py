from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.features.metadata import TrainOnlyStandardizer
from ocd_v3.features.storage import load_subject_feature_bundle


@dataclass(frozen=True)
class FeatureSchema:
    representation_names: tuple[str, ...]
    embedding_dimension: int
    metadata_dimension: int
    maximum_posts: int


def _reject_path_bearing_manifest(manifest: dict[str, Any]) -> None:
    """Refuse legacy feature manifests that embed local model filesystem paths."""
    for section_name in ("encoder_runtime", "encoder"):
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            continue
        snapshot = section.get("local_snapshot")
        if not isinstance(snapshot, dict):
            continue
        if any(
            key in snapshot
            for key in ("config_path", "model_path", "model_name_or_path")
        ):
            raise ValueError(
                "Feature manifest contains a legacy local model path; regenerate "
                "the feature set with the current path-safe identity schema"
            )


class PreparedFeatureSet:
    def __init__(self, feature_dir: Path, *, require_complete: bool = True) -> None:
        self.feature_dir = feature_dir.resolve()
        self.manifest = json.loads(
            (self.feature_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _reject_path_bearing_manifest(self.manifest)
        if require_complete and not self.manifest.get("complete_dataset"):
            raise ValueError("Training requires a complete, non-smoke feature set")
        self.bundle_dir = self.feature_dir / "subjects"
        paths = sorted(self.bundle_dir.glob("*.npz"))
        if not paths:
            raise ValueError("Feature set has no subject bundles")
        first = load_subject_feature_bundle(paths[0])
        embeddings = first["content_embeddings"]
        metadata = first["metadata"]
        self.schema = FeatureSchema(
            representation_names=tuple(str(item) for item in first["representation_names"]),
            embedding_dimension=int(embeddings.shape[2]),
            metadata_dimension=int(metadata.shape[1]),
            maximum_posts=int(
                self.manifest["post_selection"]["maximum_posts_per_subject"]
            ),
        )

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def feature_id(self) -> str:
        return str(self.manifest["feature_id"])

    def bundle_path(self, subject_id: str) -> Path:
        return self.bundle_dir / f"{subject_id}.npz"

    def load(self, subject_id: str) -> dict[str, Any]:
        path = self.bundle_path(subject_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature bundle for subject={subject_id}")
        payload = load_subject_feature_bundle(path)
        stored_subject_id = str(payload["subject_id"][0])
        if stored_subject_id != subject_id:
            raise ValueError("Feature bundle subject ID does not match its assignment")
        names = tuple(str(item) for item in payload["representation_names"])
        if names != self.schema.representation_names:
            raise ValueError("Representation schema differs between subject bundles")
        embeddings = payload["content_embeddings"]
        metadata = payload["metadata"]
        if embeddings.shape[2] != self.schema.embedding_dimension:
            raise ValueError("Embedding dimension differs between subject bundles")
        if metadata.shape[1] != self.schema.metadata_dimension:
            raise ValueError("Metadata dimension differs between subject bundles")
        if embeddings.shape[0] > self.schema.maximum_posts:
            raise ValueError("Feature bundle exceeds the configured per-subject post maximum")
        if len(set(str(item) for item in payload["post_ids"])) != embeddings.shape[0]:
            raise ValueError("Feature bundle contains duplicate post IDs")
        return payload


def fit_metadata_standardizer(
    feature_set: PreparedFeatureSet, train_subject_ids: Iterable[str]
) -> TrainOnlyStandardizer:
    rows: list[Sequence[float]] = []
    for subject_id in train_subject_ids:
        payload = feature_set.load(subject_id)
        rows.extend(payload["metadata"].tolist())
    return TrainOnlyStandardizer.fit(rows)


class UserFeatureDataset:
    def __init__(
        self,
        *,
        feature_set: PreparedFeatureSet,
        assignments: Sequence[dict[str, Any]],
        standardizer: TrainOnlyStandardizer | None,
    ) -> None:
        if not assignments:
            raise ValueError("UserFeatureDataset requires at least one subject")
        self.feature_set = feature_set
        self.assignments = tuple(sorted(assignments, key=lambda row: str(row["subject_id"])))
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        assignment = self.assignments[index]
        subject_id = str(assignment["subject_id"])
        payload = self.feature_set.load(subject_id)
        content = payload["content_embeddings"].astype(np.float32, copy=False)
        raw_metadata = payload["metadata"]
        post_count = int(content.shape[0])
        maximum_posts = self.feature_set.schema.maximum_posts
        padded_content = np.zeros(
            (
                maximum_posts,
                content.shape[1],
                content.shape[2],
            ),
            dtype=np.float32,
        )
        metadata_dimension = raw_metadata.shape[1] if self.standardizer is not None else 0
        padded_metadata = np.zeros((maximum_posts, metadata_dimension), dtype=np.float32)
        post_mask = np.zeros((maximum_posts,), dtype=np.bool_)
        if post_count:
            padded_content[:post_count] = content
            if self.standardizer is not None:
                padded_metadata[:post_count] = np.asarray(
                    [self.standardizer.transform(row) for row in raw_metadata],
                    dtype=np.float32,
                )
        post_mask[:post_count] = True
        return {
            "subject_id": subject_id,
            "label": int(assignment["label_id"]),
            "content_embeddings": torch.from_numpy(padded_content),
            "metadata": torch.from_numpy(padded_metadata),
            "post_mask": torch.from_numpy(post_mask),
            "post_count": post_count,
        }


def collate_user_features(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "subject_ids": [str(item["subject_id"]) for item in batch],
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.float32),
        "content_embeddings": torch.stack(
            [item["content_embeddings"] for item in batch], dim=0
        ),
        "metadata": torch.stack([item["metadata"] for item in batch], dim=0),
        "post_mask": torch.stack([item["post_mask"] for item in batch], dim=0),
        "post_counts": torch.tensor([int(item["post_count"]) for item in batch]),
    }