from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FEATURE_BUNDLE_SCHEMA_VERSION = 2


def save_subject_feature_bundle(
    path: Path,
    *,
    subject_id: str,
    post_ids: Sequence[str],
    metadata: Sequence[Sequence[float]],
    content_embeddings: Any,
    representation_names: Sequence[str],
    modality_presence: Sequence[Sequence[bool]],
    modality_names: Sequence[str],
    encoder_manifest: dict[str, Any],
) -> None:
    """Save numerical features without Python-object pickling.

    content_embeddings must have shape [post, representation, embedding_dimension].
    """
    import numpy as np

    post_id_array = np.asarray(list(post_ids), dtype=np.str_)
    metadata_array = np.asarray(metadata, dtype=np.float32)
    embedding_array = np.asarray(content_embeddings, dtype=np.float32)
    representation_array = np.asarray(list(representation_names), dtype=np.str_)
    modality_presence_array = np.asarray(modality_presence, dtype=np.uint8)
    modality_name_array = np.asarray(list(modality_names), dtype=np.str_)
    if len(post_id_array) == 0 and modality_presence_array.ndim == 1:
        modality_presence_array = np.empty((0, len(modality_name_array)), dtype=np.uint8)
    if metadata_array.ndim != 2:
        raise ValueError("metadata must have shape [post, feature]")
    if embedding_array.ndim != 3:
        raise ValueError(
            "content_embeddings must have shape [post, representation, embedding_dimension]"
        )
    if (
        len(post_id_array) != metadata_array.shape[0]
        or len(post_id_array) != embedding_array.shape[0]
    ):
        raise ValueError("post, metadata, and embedding row counts differ")
    if len(representation_array) != embedding_array.shape[1]:
        raise ValueError("representation_names do not match the embedding representation axis")
    if len(set(representation_array.tolist())) != len(representation_array):
        raise ValueError("representation_names must be unique")
    if modality_presence_array.ndim != 2:
        raise ValueError("modality_presence must have shape [post, modality]")
    if modality_presence_array.shape[0] != len(post_id_array):
        raise ValueError("modality_presence does not match the post axis")
    if modality_presence_array.shape[1] != len(modality_name_array):
        raise ValueError("modality_names do not match the modality axis")
    if len(set(modality_name_array.tolist())) != len(modality_name_array):
        raise ValueError("modality_names must be unique")
    if not np.isfinite(metadata_array).all() or not np.isfinite(embedding_array).all():
        raise ValueError("Feature bundles cannot contain NaN or infinite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.asarray([FEATURE_BUNDLE_SCHEMA_VERSION], dtype=np.int16),
        subject_id=np.asarray([subject_id], dtype=np.str_),
        post_ids=post_id_array,
        metadata=metadata_array,
        content_embeddings=embedding_array,
        representation_names=representation_array,
        modality_presence=modality_presence_array,
        modality_names=modality_name_array,
        encoder_manifest_json=np.asarray(
            [json.dumps(encoder_manifest, ensure_ascii=False, sort_keys=True)], dtype=np.str_
        ),
    )
    temporary.replace(path)


def load_subject_feature_bundle(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        schema_version = int(payload["schema_version"][0])
        if schema_version != FEATURE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature bundle schema: {schema_version}")
        result = {key: payload[key].copy() for key in payload.files}
    result["encoder_manifest"] = json.loads(str(result.pop("encoder_manifest_json")[0]))
    return result
