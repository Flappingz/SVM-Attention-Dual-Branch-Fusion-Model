"""Local frozen Chinese Transformer features for the text baseline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ocd_v3.config import StudyConfig
from ocd_v3.data.dataset import KEYWORD_CONDITIONS, PreparedDataset
from ocd_v3.experiments.chinese_transformer_config import (
    ChineseTransformerEncoderConfig,
)
from ocd_v3.io import write_json_atomic
from ocd_v3.provenance import git_state

CHINESE_TRANSFORMER_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EncodedPostBatch:
    embeddings: Any
    untruncated_token_counts: list[int]
    truncated: list[bool]
    empty_content_tokens: list[bool]


class PostTextEncoder(Protocol):
    hidden_size: int
    parameter_count: int

    def encode(self, texts: Sequence[str]) -> EncodedPostBatch: ...

    def manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ChineseTransformerFeatureBuildResult:
    feature_dir: Path
    manifest: dict[str, Any]


def _safe_npz_write(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class HuggingFaceChineseBertEncoder:
    """Frozen BERT-compatible encoder loaded only from a local safe checkpoint."""

    def __init__(self, model_path: Path, configuration: ChineseTransformerEncoderConfig) -> None:
        import torch
        from transformers import BertModel, BertTokenizerFast

        self.configuration = configuration
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {self.model_path}")
        if not (self.model_path / "model.safetensors").is_file():
            raise ValueError("Local model directory must contain model.safetensors")
        if configuration.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if configuration.inference_dtype == "bfloat16" and configuration.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA bfloat16 was requested but is unsupported")

        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = BertModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(configuration.device)
        self.hidden_size = int(self.model.config.hidden_size)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self._torch = torch

    def _autocast_context(self) -> Any:
        torch = self._torch
        if self.configuration.inference_dtype == "float32":
            return nullcontext()
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.configuration.inference_dtype]
        return torch.autocast(device_type=self.configuration.device, dtype=dtype)

    def encode(self, texts: Sequence[str]) -> EncodedPostBatch:
        import numpy as np

        if not texts:
            return EncodedPostBatch(
                embeddings=np.empty((0, self.hidden_size), dtype=np.float32),
                untruncated_token_counts=[],
                truncated=[],
                empty_content_tokens=[],
            )
        torch = self._torch
        all_embeddings: list[Any] = []
        all_counts: list[int] = []
        all_truncated: list[bool] = []
        all_empty: list[bool] = []
        batch_size = self.configuration.inference_batch_size
        maximum_tokens = self.configuration.maximum_tokens_per_post
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            untruncated = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
            counts = [len(token_ids) for token_ids in untruncated]
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=maximum_tokens,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = encoded.pop("special_tokens_mask").to(
                self.configuration.device
            )
            model_inputs = {
                key: value.to(self.configuration.device) for key, value in encoded.items()
            }
            with torch.inference_mode(), self._autocast_context():
                hidden = self.model(**model_inputs).last_hidden_state
            pooled, empty = mean_pool_non_special_tokens(
                hidden,
                model_inputs["attention_mask"],
                special_tokens_mask,
            )
            all_embeddings.append(pooled.float().cpu().numpy())
            all_counts.extend(counts)
            all_truncated.extend(count > maximum_tokens for count in counts)
            all_empty.extend(bool(value) for value in empty.cpu().tolist())
        return EncodedPostBatch(
            embeddings=np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False),
            untruncated_token_counts=all_counts,
            truncated=all_truncated,
            empty_content_tokens=all_empty,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.configuration.model_id,
            "revision": self.configuration.revision,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "hidden_size": self.hidden_size,
            "parameter_count": self.parameter_count,
            "weights_format": "safetensors",
            "local_files_only": True,
            "frozen": True,
        }


def mean_pool_non_special_tokens(
    hidden: Any,
    attention_mask: Any,
    special_tokens_mask: Any,
) -> tuple[Any, Any]:
    """Mean pool real content tokens and use CLS only for an empty post."""
    content_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    content_count = content_mask.sum(dim=1)
    pooled = (hidden * content_mask.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / content_count.clamp_min(1).unsqueeze(-1)
    empty = content_count == 0
    if bool(empty.any()):
        pooled[empty] = hidden[empty, 0, :]
    return pooled, empty


def _feature_identity(
    *,
    dataset_id: str,
    configuration: ChineseTransformerEncoderConfig,
    keyword_condition: str,
    keywords: Sequence[str],
    replacement: str,
    subject_limit: int | None,
    post_limit: int | None,
) -> tuple[str, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[3]
    source_git = git_state(repository)
    identity = {
        "dataset_id": dataset_id,
        "encoder_configuration": configuration.public_dict(),
        "keyword_condition": keyword_condition,
        "keyword_policy": {
            "keywords": list(keywords),
            "replacement": replacement,
            "matching": "case-insensitive substring",
        },
        "subject_limit": subject_limit,
        "post_limit": post_limit,
        "source_code": {
            "commit": source_git["commit"],
            "dirty": source_git["dirty"],
        },
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"chinese-transformer-features-{digest}", identity


def load_chinese_transformer_subject(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        schema_version = int(payload["schema_version"][0])
        if schema_version != CHINESE_TRANSFORMER_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Chinese Transformer subject schema: {path}")
        result = {name: payload[name].copy() for name in payload.files}
    embeddings = result["post_embeddings"]
    post_ids = result["post_ids"]
    if embeddings.ndim != 2 or len(embeddings) != len(post_ids):
        raise ValueError(f"Invalid Chinese Transformer feature bundle: {path}")
    if not np.isfinite(embeddings).all():
        raise FloatingPointError(f"Non-finite Chinese Transformer feature bundle: {path}")
    return result


class PreparedChineseTransformerFeatureSet:
    def __init__(self, feature_dir: Path) -> None:
        self.feature_dir = feature_dir.resolve()
        self.manifest = json.loads(
            (self.feature_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != CHINESE_TRANSFORMER_FEATURE_SCHEMA_VERSION:
            raise ValueError("Unsupported Chinese Transformer feature manifest")
        if not self.manifest.get("complete_dataset"):
            raise ValueError("Chinese Transformer feature set is incomplete")

    @property
    def feature_id(self) -> str:
        return str(self.manifest["feature_id"])

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def hidden_size(self) -> int:
        return int(self.manifest["encoder"]["hidden_size"])

    def load(self, subject_id: str) -> dict[str, Any]:
        return load_chinese_transformer_subject(
            self.feature_dir / "subjects" / f"{subject_id}.npz"
        )


def build_chinese_transformer_features(
    *,
    dataset: PreparedDataset,
    study_config: StudyConfig,
    configuration: ChineseTransformerEncoderConfig,
    model_path: Path,
    output_root: Path,
    keyword_condition: str,
    subject_limit: int | None = None,
    post_limit: int | None = None,
    encoder_factory: Callable[[Path, ChineseTransformerEncoderConfig], PostTextEncoder] = (
        HuggingFaceChineseBertEncoder
    ),
) -> ChineseTransformerFeatureBuildResult:
    if keyword_condition not in KEYWORD_CONDITIONS:
        raise ValueError("Unsupported keyword condition")
    if subject_limit is not None and subject_limit < 1:
        raise ValueError("subject_limit must be positive")
    if post_limit is not None and post_limit < 1:
        raise ValueError("post_limit must be positive")
    feature_id, identity = _feature_identity(
        dataset_id=dataset.dataset_id,
        configuration=configuration,
        keyword_condition=keyword_condition,
        keywords=study_config.dataset.keywords,
        replacement=study_config.dataset.keyword_mask,
        subject_limit=subject_limit,
        post_limit=post_limit,
    )
    feature_dir = output_root.resolve() / feature_id
    manifest_path = feature_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError("Existing feature manifest has a different identity")
        if manifest.get("complete_dataset"):
            return ChineseTransformerFeatureBuildResult(feature_dir, manifest)
    feature_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(dataset.subjects())
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    encoder: PostTextEncoder | None = None
    summaries: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.perf_counter()
    for index, subject in enumerate(subjects, start=1):
        bundle_path = feature_dir / "subjects" / f"{subject.subject_id}.npz"
        posts = dataset.selected_posts(
            subject.subject_id,
            keyword_condition=keyword_condition,
            keywords=study_config.dataset.keywords,
            replacement=study_config.dataset.keyword_mask,
        )
        if post_limit is not None:
            posts = posts[-post_limit:]
        if not posts:
            raise ValueError(f"Eligible subject has no selected posts: {subject.subject_id}")
        if bundle_path.is_file():
            payload = load_chinese_transformer_subject(bundle_path)
            token_counts = payload["untruncated_token_counts"].astype(int).tolist()
            truncated = payload["truncated"].astype(bool).tolist()
            empty_content = payload["empty_content_tokens"].astype(bool).tolist()
            embeddings = payload["post_embeddings"]
            if payload["post_ids"].astype(str).tolist() != [
                str(post["post_id"]) for post in posts
            ]:
                raise ValueError("Existing bundle no longer matches deterministic post selection")
        else:
            if encoder is None:
                encoder = encoder_factory(model_path, configuration)
            encoded = encoder.encode([str(post["cleaned_text"]) for post in posts])
            embeddings = encoded.embeddings
            token_counts = encoded.untruncated_token_counts
            truncated = encoded.truncated
            empty_content = encoded.empty_content_tokens
            if embeddings.shape != (len(posts), encoder.hidden_size):
                raise ValueError("Encoder returned an unexpected embedding shape")
            _safe_npz_write(
                bundle_path,
                schema_version=[CHINESE_TRANSFORMER_FEATURE_SCHEMA_VERSION],
                subject_id=[subject.subject_id],
                post_ids=[str(post["post_id"]) for post in posts],
                post_embeddings=embeddings,
                untruncated_token_counts=token_counts,
                truncated=truncated,
                empty_content_tokens=empty_content,
            )
        counters["subjects"] += 1
        counters["posts"] += len(posts)
        counters["truncated_posts"] += sum(bool(value) for value in truncated)
        counters["empty_content_token_posts"] += sum(bool(value) for value in empty_content)
        counters["tokens_before_truncation"] += sum(int(value) for value in token_counts)
        summaries.append(
            {
                "subject_id": subject.subject_id,
                "label_id": subject.label_id,
                "selected_post_count": len(posts),
                "truncated_post_count": sum(bool(value) for value in truncated),
                "empty_content_token_post_count": sum(bool(value) for value in empty_content),
            }
        )
        write_json_atomic(
            feature_dir / "progress.json",
            {
                "feature_id": feature_id,
                "status": "running",
                "completed_subjects": index,
                "total_subjects": len(subjects),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if encoder is None:
        # A resumed build still needs an authoritative local encoder manifest.
        encoder = encoder_factory(model_path, configuration)
    encoder_manifest = encoder.manifest()
    complete_dataset = subject_limit is None and post_limit is None
    manifest = {
        "schema_version": CHINESE_TRANSFORMER_FEATURE_SCHEMA_VERSION,
        "feature_id": feature_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset.dataset_id,
        "identity": identity,
        "encoder_configuration": configuration.public_dict(),
        "encoder": encoder_manifest,
        "keyword_condition": keyword_condition,
        "keyword_policy": {
            "keywords": list(study_config.dataset.keywords),
            "replacement": study_config.dataset.keyword_mask,
            "matching": "case-insensitive substring",
        },
        "pooling": {
            "tokens": configuration.token_pooling,
            "posts": configuration.post_pooling,
            "empty_content_fallback": "CLS representation",
        },
        "complete_dataset": complete_dataset,
        "subject_limit": subject_limit,
        "post_limit": post_limit,
        "counts": dict(sorted(counters.items())),
        "git": git_state(Path(__file__).resolve().parents[3]),
    }
    write_json_atomic(feature_dir / "subjects_summary.json", summaries)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        feature_dir / "progress.json",
        {
            "feature_id": feature_id,
            "status": "complete",
            "completed_subjects": len(subjects),
            "total_subjects": len(subjects),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return ChineseTransformerFeatureBuildResult(feature_dir, manifest)
