from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import KEYWORD_CONDITIONS, PreparedDataset
from ocd_v3.data.private_mapping import load_private_media_paths
from ocd_v3.experiments.full_config import FullExperimentConfig
from ocd_v3.features.qwen_vl import PostEmbeddingInput, Qwen3VLEmbedder, evenly_sample_paths
from ocd_v3.features.storage import (
    load_subject_feature_bundle,
    save_subject_feature_bundle,
)
from ocd_v3.features.training_data import PreparedFeatureSet
from ocd_v3.io import write_json_atomic
from ocd_v3.provenance import directory_content_fingerprint, git_state

FEATURE_SET_SCHEMA_VERSION = 6
MODALITY_NAMES = ("text", "image", "live_photo")
EMBEDDING_INPUT_MODALITIES = frozenset({"text", "image"})


def normalize_embedding_input_modalities(values: tuple[str, ...]) -> tuple[str, ...]:
    modalities = tuple(str(value).strip() for value in values)
    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("embedding input modalities must be a non-empty unique sequence")
    unknown = set(modalities) - EMBEDDING_INPUT_MODALITIES
    if unknown:
        raise ValueError(f"Unknown embedding input modalities: {sorted(unknown)}")
    return tuple(name for name in ("text", "image") if name in modalities)


class PostEmbedder(Protocol):
    hidden_size: int

    def embed_post(self, value: PostEmbeddingInput) -> Any: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PostMediaSelection:
    available_images: int
    selected_images: int
    available_live_photos: int
    selected_live_photos: int


@dataclass(frozen=True)
class PostSelectionAudit:
    available_post_count: int
    full_history_keyword_removed_post_count: int
    post_count_after_keyword_removal: int
    model_window_keyword_removed_post_count: int
    selected_post_count: int


def _encoder_runtime_identity(model_path: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "safetensors", "av"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "local_snapshot": directory_content_fingerprint(model_path),
        "packages": packages,
    }


def _feature_identity(
    dataset_id: str,
    study_config: StudyConfig,
    full_config: FullExperimentConfig,
    keyword_condition: str,
    encoder_runtime_identity: dict[str, Any],
    reused_feature_id: str | None,
    subject_limit: int | None,
    post_limit: int | None,
    embedding_input_modalities: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[3]
    source_git = git_state(repository)
    identity = {
        "schema_version": FEATURE_SET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "full_config": full_config.public_dict(),
        "keyword_condition": keyword_condition,
        "encoder_runtime": encoder_runtime_identity,
        "keyword_policy": {
            "keywords": list(study_config.dataset.keywords),
            "replacement": study_config.dataset.keyword_mask,
        },
        "reused_feature_id": reused_feature_id,
        "subject_limit": subject_limit,
        "post_limit": post_limit,
        "embedding_input_modalities": list(embedding_input_modalities),
        "source_code": {
            "commit": source_git["commit"],
            "dirty": source_git["dirty"],
            "worktree_sha256": source_git["worktree_sha256"],
        },
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"features-{digest}", identity


def _media_by_post(
    dataset: PreparedDataset,
    study_config: StudyConfig,
    subject_id: str,
) -> dict[str, dict[str, list[Path]]]:
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: {"image": [], "live_photo": []}
    )
    private_media_paths = load_private_media_paths(
        study_config, dataset.dataset_id, subject_id
    )
    for row in dataset.media(subject_id):
        post_id = row.get("post_id")
        kind = str(row.get("kind"))
        if not post_id or kind not in {"image", "live_photo"}:
            continue
        media_id = str(row.get("media_id", ""))
        if media_id not in private_media_paths:
            raise ValueError("Prepared media ID is absent from the private identity mapping")
        path = private_media_paths[media_id]
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared media is missing for anonymous subject={subject_id}, post={post_id}"
            )
        grouped[str(post_id)][kind].append(path)
    for values in grouped.values():
        values["image"].sort(key=lambda item: item.name)
        values["live_photo"].sort(key=lambda item: item.name)
    return grouped


def build_subject_inputs(
    dataset: PreparedDataset,
    study_config: StudyConfig,
    full_config: FullExperimentConfig,
    subject_id: str,
    *,
    keyword_condition: str,
    embedding_input_modalities: tuple[str, ...] = ("text", "image"),
    post_limit: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[PostEmbeddingInput],
    list[list[bool]],
    list[PostMediaSelection],
    PostSelectionAudit,
]:
    embedding_input_modalities = normalize_embedding_input_modalities(
        embedding_input_modalities
    )
    available_posts = dataset.posts(subject_id)
    post_count_after_keyword_removal = len(available_posts)
    full_history_keyword_removed_post_count = 0
    if keyword_condition == "removed":
        post_count_after_keyword_removal = len(
            dataset.posts(
                subject_id,
                keyword_condition="removed",
                keywords=study_config.dataset.keywords,
            )
        )
        full_history_keyword_removed_post_count = (
            len(available_posts) - post_count_after_keyword_removal
        )
    model_window_posts = dataset.selected_posts(subject_id)
    posts = dataset.selected_posts(
        subject_id,
        keyword_condition=keyword_condition,
        keywords=study_config.dataset.keywords,
        replacement=study_config.dataset.keyword_mask,
    )
    if post_limit is not None:
        if post_limit < 1:
            raise ValueError("post_limit must be positive")
        posts = posts[-post_limit:]
    if not posts and keyword_condition != "removed":
        raise ValueError(f"Eligible anonymous subject has no selected posts: {subject_id}")
    grouped_media = _media_by_post(dataset, study_config, subject_id)
    values: list[PostEmbeddingInput] = []
    presence: list[list[bool]] = []
    media_selections: list[PostMediaSelection] = []
    for post in posts:
        media = grouped_media.get(str(post["post_id"]), {"image": [], "live_photo": []})
        images = (
            evenly_sample_paths(media["image"], full_config.encoder.max_images_per_post)
            if "image" in embedding_input_modalities
            else ()
        )
        videos = (
            evenly_sample_paths(
                media["live_photo"], full_config.encoder.max_dynamic_media_per_post
            )
            if "image" in embedding_input_modalities
            else ()
        )
        text = (
            str(post["cleaned_text"])
            if "text" in embedding_input_modalities
            else ""
        )
        values.append(PostEmbeddingInput(text=text, images=images, videos=videos))
        presence.append([bool(text.strip()), bool(images), bool(videos)])
        media_selections.append(
            PostMediaSelection(
                available_images=len(media["image"]),
                selected_images=len(images),
                available_live_photos=len(media["live_photo"]),
                selected_live_photos=len(videos),
            )
        )
    return (
        posts,
        values,
        presence,
        media_selections,
        PostSelectionAudit(
            available_post_count=len(available_posts),
            full_history_keyword_removed_post_count=(
                full_history_keyword_removed_post_count
            ),
            post_count_after_keyword_removal=post_count_after_keyword_removal,
            model_window_keyword_removed_post_count=(len(model_window_posts) - len(posts)),
            selected_post_count=len(posts),
        ),
    )


def build_feature_set(
    *,
    dataset: PreparedDataset,
    study_config: StudyConfig,
    full_config: FullExperimentConfig,
    keyword_condition: str,
    output_root: Path,
    subject_limit: int | None = None,
    post_limit: int | None = None,
    source_feature_set: PreparedFeatureSet | None = None,
    embedding_input_modalities: tuple[str, ...] = ("text", "image"),
    embedder_factory: Callable[[Any], PostEmbedder] = Qwen3VLEmbedder,
) -> FeatureBuildResult:
    if keyword_condition not in KEYWORD_CONDITIONS:
        raise ValueError("keyword_condition must be original, masked, or removed")
    if subject_limit is not None and subject_limit < 1:
        raise ValueError("subject_limit must be positive")
    embedding_input_modalities = normalize_embedding_input_modalities(
        embedding_input_modalities
    )
    if source_feature_set is not None:
        if keyword_condition != "removed":
            raise ValueError("A source feature set can be used only for keyword_condition=removed")
        if source_feature_set.dataset_id != dataset.dataset_id:
            raise ValueError("Source feature set refers to a different dataset")
        if not source_feature_set.manifest.get("complete_dataset"):
            raise ValueError("Source feature set must cover the complete dataset")
        if source_feature_set.manifest.get("keyword_condition") != "original":
            raise ValueError("Source feature set must use keyword_condition=original")
        source_modalities = tuple(
            source_feature_set.manifest.get(
                "embedding_input_modalities", ("text", "image")
            )
        )
        if source_modalities != embedding_input_modalities:
            raise ValueError(
                "Source feature embedding modalities differ from the requested condition"
            )
        if source_feature_set.schema.representation_names != full_config.encoder.representations:
            raise ValueError("Source feature representations differ from the requested encoder")
        if source_feature_set.schema.maximum_posts != (
            study_config.dataset.post_selection.maximum_posts_per_subject
        ):
            raise ValueError("Source feature post cap differs from the study configuration")
        source_encoder = source_feature_set.manifest.get("encoder")
        if not isinstance(source_encoder, dict):
            raise ValueError("Source feature manifest has no encoder record")
        for field, expected in (
            ("model_id", full_config.encoder.model_id),
            ("revision", full_config.encoder.revision),
            ("normalize", full_config.encoder.normalize),
            ("instruction", full_config.encoder.instruction),
        ):
            if source_encoder.get(field) != expected:
                raise ValueError(
                    f"Source feature encoder {field} differs from the requested encoder"
                )
    encoder_runtime_identity = (
        {
            "local_snapshot": source_encoder.get("local_snapshot"),
            "packages": source_encoder.get("packages"),
        }
        if source_feature_set is not None
        else _encoder_runtime_identity(full_config.encoder.model_name_or_path)
    )
    feature_id, identity = _feature_identity(
        dataset.dataset_id,
        study_config,
        full_config,
        keyword_condition,
        encoder_runtime_identity,
        source_feature_set.feature_id if source_feature_set is not None else None,
        subject_limit,
        post_limit,
        embedding_input_modalities,
    )
    feature_dir = output_root.resolve() / feature_id
    bundle_dir = feature_dir / "subjects"
    manifest_path = feature_dir / "manifest.json"
    if manifest_path.is_file():
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_identity = {key: stored_manifest.get(key) for key in identity}
        if stored_manifest.get("feature_id") != feature_id or stored_identity != identity:
            raise RuntimeError("Cached feature manifest does not match the requested identity")
        return FeatureBuildResult(feature_dir=feature_dir, manifest=stored_manifest)

    subjects = sorted(dataset.subjects(), key=lambda item: item.subject_id)
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    embedder: PostEmbedder | None = None
    encoder_manifest: dict[str, Any] | None = (
        dict(source_feature_set.manifest["encoder"])
        if source_feature_set is not None
        else None
    )
    encoder_load_seconds = 0.0
    embedding_seconds = 0.0
    embedded_posts_this_invocation = 0
    resumed_subjects = 0
    counters: Counter[str] = Counter()
    subject_summaries: list[dict[str, Any]] = []
    empty_content_embedding: Any | None = None
    bundle_dir.mkdir(parents=True, exist_ok=True)
    invocation_started = time.perf_counter()

    for subject in subjects:
        posts, inputs, presence, media_selections, selection_audit = build_subject_inputs(
            dataset,
            study_config,
            full_config,
            subject.subject_id,
            keyword_condition=keyword_condition,
            embedding_input_modalities=embedding_input_modalities,
            post_limit=post_limit,
        )
        post_ids = [str(post["post_id"]) for post in posts]
        bundle_path = bundle_dir / f"{subject.subject_id}.npz"
        if bundle_path.is_file():
            resumed_subjects += 1
            payload = load_subject_feature_bundle(bundle_path)
            if [str(value) for value in payload["post_ids"]] != post_ids:
                raise ValueError(
                    f"Partial feature bundle post order differs for subject={subject.subject_id}"
                )
            if tuple(str(value) for value in payload["representation_names"]) != tuple(
                full_config.encoder.representations
            ):
                raise ValueError("Partial feature bundle representation schema differs")
            if payload["modality_presence"].tolist() != [
                [int(value) for value in row] for row in presence
            ]:
                raise ValueError("Partial feature bundle modality state differs")
            stored_encoder_manifest = payload["encoder_manifest"]
            if encoder_manifest is None:
                encoder_manifest = stored_encoder_manifest
            elif encoder_manifest != stored_encoder_manifest:
                raise ValueError("Partial bundles were created by different encoders")
        else:
            source_payload: dict[str, Any] | None = None
            if source_feature_set is not None:
                source_payload = source_feature_set.load(subject.subject_id)
                source_post_indices = {
                    str(post_id): index
                    for index, post_id in enumerate(source_payload["post_ids"])
                }
                missing_post_ids = [
                    post_id
                    for post_id in post_ids
                    if post_id not in source_post_indices
                ]
                if missing_post_ids:
                    raise ValueError(
                        "Keyword removal selected posts absent from the original source "
                        f"feature bundle for subject={subject.subject_id}"
                    )
            if embedder is None:
                if source_payload is None:
                    load_started = time.perf_counter()
                    embedder = embedder_factory(full_config.encoder)
                    embedder._snapshot_identity = encoder_runtime_identity["local_snapshot"]
                    encoder_load_seconds += time.perf_counter() - load_started
                    current_manifest = embedder.manifest()
                    current_manifest["local_snapshot"] = encoder_runtime_identity[
                        "local_snapshot"
                    ]
                    current_manifest["packages"] = encoder_runtime_identity["packages"]
                    if encoder_manifest is not None and encoder_manifest != current_manifest:
                        raise ValueError("Current encoder differs from partial feature bundles")
                    encoder_manifest = current_manifest
            try:
                if source_payload is not None:
                    embeddings = source_payload["content_embeddings"][
                        [source_post_indices[post_id] for post_id in post_ids]
                    ]
                    counters["reused_source_feature_posts"] += len(post_ids)
                elif inputs:
                    assert embedder is not None
                    embeddings = []
                    for value in inputs:
                        is_empty_content = (
                            not value.text and not value.images and not value.videos
                        )
                        if is_empty_content and empty_content_embedding is not None:
                            embeddings.append(empty_content_embedding.copy())
                            counters["reused_empty_content_placeholder_embeddings"] += 1
                        else:
                            embedding_started = time.perf_counter()
                            embedding = embedder.embed_post(value)
                            embeddings.append(embedding)
                            embedding_seconds += time.perf_counter() - embedding_started
                            embedded_posts_this_invocation += 1
                            if is_empty_content:
                                empty_content_embedding = embedding.copy()
                else:
                    import numpy as np

                    assert embedder is not None
                    embeddings = np.empty(
                        (
                            0,
                            len(full_config.encoder.representations),
                            embedder.hidden_size,
                        ),
                        dtype=np.float32,
                    )
            except Exception as error:
                write_json_atomic(
                    feature_dir / "progress.json",
                    {
                        "schema_version": 1,
                        "feature_id": feature_id,
                        "completed_subject_ids": [
                            row["subject_id"] for row in subject_summaries
                        ],
                        "failed_subject_id": subject.subject_id,
                        "error_type": type(error).__name__,
                    },
                )
                raise
            if posts:
                metadata = [post["metadata_vector"] for post in posts]
            else:
                import numpy as np

                metadata = np.empty(
                    (0, len(dataset.manifest["metadata_features"])), dtype=np.float32
                )
            save_subject_feature_bundle(
                bundle_path,
                subject_id=subject.subject_id,
                post_ids=post_ids,
                metadata=metadata,
                content_embeddings=embeddings,
                representation_names=full_config.encoder.representations,
                modality_presence=presence,
                modality_names=MODALITY_NAMES,
                encoder_manifest=encoder_manifest,
            )
        modality_counts = {
            name: sum(row[index] for row in presence)
            for index, name in enumerate(MODALITY_NAMES)
        }
        counters["subjects"] += 1
        counters["posts"] += len(posts)
        counters["keyword_matched_posts_removed_from_full_history"] += (
            selection_audit.full_history_keyword_removed_post_count
        )
        counters["keyword_matched_posts_removed_from_model_window"] += (
            selection_audit.model_window_keyword_removed_post_count
        )
        counters["subjects_with_no_posts_in_removed_model_window"] += int(
            selection_audit.selected_post_count == 0
        )
        counters.update({f"posts_with_{name}": count for name, count in modality_counts.items()})
        counters["available_images"] += sum(
            item.available_images for item in media_selections
        )
        counters["selected_images"] += sum(
            item.selected_images for item in media_selections
        )
        counters["dropped_images_by_cap"] += sum(
            item.available_images - item.selected_images for item in media_selections
        )
        counters["posts_truncated_by_image_cap"] += sum(
            item.available_images > item.selected_images for item in media_selections
        )
        counters["available_live_photos"] += sum(
            item.available_live_photos for item in media_selections
        )
        counters["selected_live_photos"] += sum(
            item.selected_live_photos for item in media_selections
        )
        counters["dropped_live_photos_by_cap"] += sum(
            item.available_live_photos - item.selected_live_photos
            for item in media_selections
        )
        counters["posts_truncated_by_live_photo_cap"] += sum(
            item.available_live_photos > item.selected_live_photos
            for item in media_selections
        )
        counters["selected_visual_frames"] += sum(
            item.selected_images
            + item.selected_live_photos * full_config.encoder.max_frames
            for item in media_selections
        )
        subject_summaries.append(
            {
                "subject_id": subject.subject_id,
                "label_id": subject.label_id,
                "available_post_count": selection_audit.available_post_count,
                "keyword_matched_posts_removed_from_full_history": (
                    selection_audit.full_history_keyword_removed_post_count
                ),
                "post_count_after_keyword_removal": (
                    selection_audit.post_count_after_keyword_removal
                ),
                "keyword_matched_posts_removed_from_model_window": (
                    selection_audit.model_window_keyword_removed_post_count
                ),
                "post_count": len(posts),
                "modality_counts": modality_counts,
            }
        )
        write_json_atomic(
            feature_dir / "progress.json",
            {
                "schema_version": 1,
                "feature_id": feature_id,
                "completed_subject_ids": [row["subject_id"] for row in subject_summaries],
            },
        )

    if encoder_manifest is None:
        raise RuntimeError("Feature construction produced no encoder manifest")
    expected_subjects = list(dataset.subjects())
    complete = (
        subject_limit is None
        and post_limit is None
        and len(subjects) == len(expected_subjects)
    )
    manifest = {
        **identity,
        "feature_id": feature_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete_dataset": complete,
        "post_selection": {
            "maximum_posts_per_subject": (
                study_config.dataset.post_selection.maximum_posts_per_subject
            ),
            "strategy": study_config.dataset.post_selection.strategy,
            "smoke_post_limit": post_limit,
        },
        "keyword_post_removal": {
            "active": keyword_condition == "removed",
            "rule": (
                "within the deterministic most_recent model-input window, exclude "
                "complete posts whose cleaned_text contains any configured keyword"
            ),
            "users_without_posts_are_retained": True,
        },
        "embedding_derivation": (
            {
                "source_feature_id": source_feature_set.feature_id,
                "operation": (
                    "reuse original-condition post embeddings after removing a subset "
                    "of the fixed model-input post window"
                ),
            }
            if source_feature_set is not None
            else None
        ),
        "embedding_input_modalities": list(embedding_input_modalities),
        "visual_modality_definition": "static images and Live Photo video frames",
        "missing_input_policy": (
            "retain every condition post; absent selected content is represented by the "
            "encoder's literal NULL text placeholder"
        ),
        "modalities": list(MODALITY_NAMES) + ["metadata"],
        "metadata_features": list(dataset.manifest["metadata_features"]),
        "encoder": encoder_manifest,
        "counts": dict(sorted(counters.items())),
        "build_runtime": {
            "invocation_wall_seconds": time.perf_counter() - invocation_started,
            "encoder_load_seconds": encoder_load_seconds,
            "embedding_seconds": embedding_seconds,
            "embedded_posts_this_invocation": embedded_posts_this_invocation,
            "resumed_subjects": resumed_subjects,
        },
    }
    write_json_atomic(
        feature_dir / "subjects_summary.json",
        subject_summaries,
    )
    write_json_atomic(manifest_path, manifest)
    return FeatureBuildResult(feature_dir=feature_dir, manifest=manifest)
