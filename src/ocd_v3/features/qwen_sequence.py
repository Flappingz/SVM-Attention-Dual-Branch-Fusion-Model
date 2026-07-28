"""Deterministic user-level summaries of ordered Qwen post embeddings."""

from __future__ import annotations

QWEN_SEQUENCE_BLOCKS = (
    "global_mean",
    "global_std",
    "early_half_mean",
    "late_half_mean",
)

QWEN_TEMPORAL_PYRAMID_BLOCKS = (
    "global_mean",
    "early_half_mean",
    "late_half_mean",
)

QWEN_SEGMENT_MEAN_STD2_BLOCKS = (
    "global_mean",
    "global_std",
    "early_half_mean",
    "late_half_mean",
    "early_half_std",
    "late_half_std",
)


def _l2_normalize(vector: object) -> object:
    import numpy as np

    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def qwen_mean_std_temporal_halves(
    post_embeddings: object,
    *,
    expected_embedding_dimension: int,
) -> object:
    """Encode an ordered post sequence as normalized mean/dispersion/time blocks.

    The sequence is assumed to be in chronological order.  Population standard
    deviation (ddof=0) is used so a single-post sequence remains finite.
    """
    import numpy as np

    values = np.asarray(post_embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_embedding_dimension:
        raise ValueError(
            "post_embeddings must have shape [post, expected_embedding_dimension]"
        )
    if values.shape[0] < 1:
        raise ValueError("A user sequence must retain at least one post")
    if not np.isfinite(values).all():
        raise ValueError("Qwen sequence contains non-finite embeddings")

    post_count = int(values.shape[0])
    middle = max(1, post_count // 2)
    early = values[:middle]
    late = values[middle:] if middle < post_count else values[-1:]
    blocks = (
        values.mean(axis=0, dtype=np.float64),
        values.std(axis=0, dtype=np.float64),
        early.mean(axis=0, dtype=np.float64),
        late.mean(axis=0, dtype=np.float64),
    )
    result = np.concatenate([_l2_normalize(block) for block in blocks]).astype(
        np.float64, copy=False
    )
    if result.shape != (4 * expected_embedding_dimension,):
        raise AssertionError("Unexpected Qwen sequence-summary shape")
    if not np.isfinite(result).all():
        raise FloatingPointError("Qwen sequence summary is non-finite")
    return result


def qwen_temporal_pyramid2(
    post_embeddings: object,
    *,
    expected_embedding_dimension: int,
) -> object:
    """Encode the ordered sequence with global, early-half, and late-half means."""
    import numpy as np

    values = np.asarray(post_embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_embedding_dimension:
        raise ValueError(
            "post_embeddings must have shape [post, expected_embedding_dimension]"
        )
    if values.shape[0] < 1:
        raise ValueError("A user sequence must retain at least one post")
    if not np.isfinite(values).all():
        raise ValueError("Qwen sequence contains non-finite embeddings")
    middle = max(1, int(values.shape[0]) // 2)
    early = values[:middle]
    late = values[middle:] if middle < values.shape[0] else values[-1:]
    blocks = (
        values.mean(axis=0, dtype=np.float64),
        early.mean(axis=0, dtype=np.float64),
        late.mean(axis=0, dtype=np.float64),
    )
    result = np.concatenate([_l2_normalize(block) for block in blocks]).astype(
        np.float64, copy=False
    )
    if result.shape != (3 * expected_embedding_dimension,):
        raise AssertionError("Unexpected Qwen temporal-pyramid shape")
    if not np.isfinite(result).all():
        raise FloatingPointError("Qwen temporal-pyramid summary is non-finite")
    return result


def qwen_segment_mean_std2(
    post_embeddings: object,
    *,
    expected_embedding_dimension: int,
) -> object:
    """Encode global and temporal-half location/dispersion statistics."""
    import numpy as np

    values = np.asarray(post_embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_embedding_dimension:
        raise ValueError(
            "post_embeddings must have shape [post, expected_embedding_dimension]"
        )
    if values.shape[0] < 1:
        raise ValueError("A user sequence must retain at least one post")
    if not np.isfinite(values).all():
        raise ValueError("Qwen sequence contains non-finite embeddings")
    middle = max(1, int(values.shape[0]) // 2)
    early = values[:middle]
    late = values[middle:] if middle < values.shape[0] else values[-1:]
    blocks = (
        values.mean(axis=0, dtype=np.float64),
        values.std(axis=0, dtype=np.float64),
        early.mean(axis=0, dtype=np.float64),
        late.mean(axis=0, dtype=np.float64),
        early.std(axis=0, dtype=np.float64),
        late.std(axis=0, dtype=np.float64),
    )
    result = np.concatenate([_l2_normalize(block) for block in blocks]).astype(
        np.float64, copy=False
    )
    if result.shape != (6 * expected_embedding_dimension,):
        raise AssertionError("Unexpected Qwen segment-moment summary shape")
    if not np.isfinite(result).all():
        raise FloatingPointError("Qwen segment-moment summary is non-finite")
    return result
