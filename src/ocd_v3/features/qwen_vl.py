from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocd_v3.experiments.full_config import EncoderConfig
from ocd_v3.provenance import directory_content_fingerprint


@dataclass(frozen=True)
class PostEmbeddingInput:
    text: str
    images: tuple[Path, ...]
    videos: tuple[Path, ...]


class LocalMediaDecodeError(RuntimeError):
    """Raised when a local image or video cannot be decoded."""


def configured_attention_implementation() -> str | None:
    """Return the explicitly requested Transformers attention implementation.

    The default stays under Transformers control.  A concrete implementation is
    intentionally opt-in through the environment so a feature manifest can
    distinguish ordinary eager/SDPA builds from a FlashAttention-2 build.
    """

    value = os.environ.get("QWEN3_VL_ATTN_IMPLEMENTATION")
    if value is None or not value.strip():
        return None
    value = value.strip()
    if value not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("QWEN3_VL_ATTN_IMPLEMENTATION must be eager, sdpa, or flash_attention_2")
    return value


def evenly_sample_paths(paths: Sequence[Path], maximum: int) -> tuple[Path, ...]:
    """Select a deterministic, order-preserving spread without reading media."""
    ordered = tuple(paths)
    if maximum < 0:
        raise ValueError("maximum cannot be negative")
    if maximum == 0 or not ordered:
        return ()
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return (ordered[0],)
    indices = [round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)]
    return tuple(ordered[index] for index in indices)


def load_image_copy(path: Path) -> Any:
    from PIL import Image

    try:
        with Image.open(path) as source:
            return source.convert("RGB").copy()
    except Exception as error:
        raise LocalMediaDecodeError(
            f"Failed to decode image {path} ({type(error).__name__})"
        ) from None


def _decode_video_frames(path: Path, maximum_frames: int) -> list[Any]:
    """Decode a bounded, deterministic set of frames with PyAV.

    qwen-vl-utils 0.0.14 falls back to torchvision.io.read_video, which was
    removed from torchvision 0.28. Decoding locally also ensures that only the
    selected frame budget enters the processor.
    """
    import av

    if maximum_frames < 2:
        raise ValueError("Qwen video inputs require at least two frames")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Dynamic media has no video stream")
        stream = container.streams.video[0]
        declared_frames = int(stream.frames or 0)
        target_indices: list[int] | None = None
        target_pts: list[int] | None = None
        if declared_frames > 0:
            target_count = min(maximum_frames, declared_frames)
            if target_count == 1:
                target_indices = [0]
            else:
                target_indices = [
                    round(index * (declared_frames - 1) / (target_count - 1))
                    for index in range(target_count)
                ]
        elif stream.duration is not None:
            target_count = maximum_frames
            start = int(stream.start_time or 0)
            stop = start + max(int(stream.duration) - 1, 0)
            target_pts = [
                round(start + index * (stop - start) / (target_count - 1))
                for index in range(target_count)
            ]

        selected: list[Any] = []
        fallback_first: Any | None = None
        fallback_last: Any | None = None
        next_target = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            image = frame.to_image()
            if fallback_first is None:
                fallback_first = image
            fallback_last = image
            if target_indices is not None:
                while (
                    next_target < len(target_indices) and frame_index >= target_indices[next_target]
                ):
                    selected.append(image.copy())
                    next_target += 1
            elif target_pts is not None and frame.pts is not None:
                while next_target < len(target_pts) and int(frame.pts) >= target_pts[next_target]:
                    selected.append(image.copy())
                    next_target += 1
            if next_target >= maximum_frames:
                break
        if fallback_first is None or fallback_last is None:
            raise ValueError("Dynamic media contains no decodable frames")
        if not selected:
            selected = [fallback_first, fallback_last]
        while len(selected) < 2:
            selected.append(fallback_last.copy())
        while len(selected) < maximum_frames:
            selected.append(fallback_last.copy())
        return selected[:maximum_frames]


def decode_video_frames(path: Path, maximum_frames: int) -> list[Any]:
    try:
        return _decode_video_frames(path, maximum_frames)
    except LocalMediaDecodeError:
        raise
    except Exception as error:
        raise LocalMediaDecodeError(
            f"Failed to decode dynamic media {path} ({type(error).__name__})"
        ) from None


class Qwen3VLEmbedder:
    """Thin, inference-only wrapper around the official Qwen3-VL embedding recipe."""

    def __init__(self, config: EncoderConfig) -> None:
        import torch
        import torch.nn.functional as functional
        from qwen_vl_utils.vision_process import process_vision_info
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLPreTrainedModel,
        )
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            _checkpoint_conversion_mapping: dict[str, str] = {}
            accepts_loss_kwargs = False

            def __init__(self, model_config: Any) -> None:
                super().__init__(model_config)
                self.model = Qwen3VLModel(model_config)
                self.post_init()

        if not config.model_name_or_path.is_dir():
            raise FileNotFoundError("Configured Qwen3-VL model directory does not exist")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("encoder.device=cuda but CUDA is unavailable")
        device_name = (
            "cuda"
            if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        attention_implementation = configured_attention_implementation()
        if attention_implementation is not None:
            load_kwargs["attn_implementation"] = attention_implementation
        if config.torch_dtype != "auto":
            load_kwargs["dtype"] = dtype_by_name[config.torch_dtype]
        self.model = Qwen3VLForEmbedding.from_pretrained(
            str(config.model_name_or_path), **load_kwargs
        ).to(torch.device(device_name))
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            str(config.model_name_or_path), padding_side="right"
        )
        self.config = config
        self._torch = torch
        self._functional = functional
        self._process_vision_info = process_vision_info
        self.device = torch.device(device_name)
        self.attention_implementation = attention_implementation

        text_config = self.model.config.text_config
        self.hidden_size = int(text_config.hidden_size)
        self.transformer_layer_count = int(text_config.num_hidden_layers)
        self._validate_representations()

    def _validate_representations(self) -> None:
        selected_layers: set[int] = set()
        for name in self.config.representations:
            if name == "final":
                continue
            layer = int(name.split(":", 1)[1])
            if layer > self.transformer_layer_count:
                raise ValueError(
                    f"Requested {name}, but encoder has {self.transformer_layer_count} layers"
                )
            selected_layers.add(layer)
        if (
            "final" in self.config.representations
            and self.transformer_layer_count in selected_layers
        ):
            raise ValueError("'final' and the final explicit transformer layer are duplicates")

    def _conversation(self, value: PostEmbeddingInput) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for video in value.videos:
            content.append(
                {
                    "type": "video",
                    "video": decode_video_frames(video, self.config.max_frames),
                    "sample_fps": self.config.fps,
                    "total_pixels": self.config.total_pixels,
                }
            )
        for image in value.images:
            content.append(
                {
                    "type": "image",
                    "image": load_image_copy(image),
                    "min_pixels": self.config.min_pixels,
                    "max_pixels": self.config.max_pixels,
                }
            )
        content.append({"type": "text", "text": value.text or "NULL"})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.instruction}],
            },
            {"role": "user", "content": content},
        ]

    def _pool_last(self, hidden_state: Any, attention_mask: Any) -> Any:
        flipped = attention_mask.flip(dims=[1])
        columns = attention_mask.shape[1] - flipped.argmax(dim=1) - 1
        rows = self._torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[rows, columns]

    def embed_post(self, value: PostEmbeddingInput) -> Any:
        conversation = self._conversation(value)
        rendered = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False
        )
        images, video_inputs, video_kwargs = self._process_vision_info(
            [conversation],
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs, strict=True)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        inputs = self.processor(
            text=rendered,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        needs_hidden_states = any(name != "final" for name in self.config.representations)
        with self._torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_hidden_states=needs_hidden_states,
                return_dict=True,
            )
            vectors: list[Any] = []
            for name in self.config.representations:
                hidden_state = (
                    outputs.last_hidden_state
                    if name == "final"
                    else outputs.hidden_states[int(name.split(":", 1)[1])]
                )
                vector = self._pool_last(hidden_state, inputs["attention_mask"])
                if self.config.normalize:
                    vector = self._functional.normalize(vector, p=2, dim=-1)
                vectors.append(vector[0].float().cpu())
        return self._torch.stack(vectors, dim=0).numpy()

    def embed_text_batch(self, texts: Sequence[str]) -> Any:
        """Embed a batch of text-only posts with the same chat template and pooling."""
        if not texts:
            raise ValueError("embed_text_batch requires at least one text")
        conversations = [
            self._conversation(PostEmbeddingInput(text=text, images=(), videos=()))
            for text in texts
        ]
        rendered = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=rendered,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        needs_hidden_states = any(name != "final" for name in self.config.representations)
        with self._torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_hidden_states=needs_hidden_states,
                return_dict=True,
            )
            vectors: list[Any] = []
            for name in self.config.representations:
                hidden_state = (
                    outputs.last_hidden_state
                    if name == "final"
                    else outputs.hidden_states[int(name.split(":", 1)[1])]
                )
                vector = self._pool_last(hidden_state, inputs["attention_mask"])
                if self.config.normalize:
                    vector = self._functional.normalize(vector, p=2, dim=-1)
                vectors.append(vector.float().cpu())
        return self._torch.stack(vectors, dim=1).numpy()

    def manifest(self) -> dict[str, Any]:
        model_path = self.config.model_name_or_path
        snapshot_identity = getattr(self, "_snapshot_identity", None)
        if snapshot_identity is None:
            snapshot_identity = directory_content_fingerprint(model_path)
        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "representations": list(self.config.representations),
            "representation_semantics": {
                "final": "MRL-compatible final Qwen embedding pooled at the last valid token",
                "layer:N": "one-based transformer block output pooled at the last valid token",
            },
            "hidden_size": self.hidden_size,
            "transformer_layer_count": self.transformer_layer_count,
            "normalize": self.config.normalize,
            "instruction": self.config.instruction,
            "torch_dtype": self.config.torch_dtype,
            "attention_implementation": self.attention_implementation,
            "max_length": self.config.max_length,
            "vision": {
                "min_pixels": self.config.min_pixels,
                "max_pixels": self.config.max_pixels,
                "total_pixels": self.config.total_pixels,
                "fps": self.config.fps,
                "max_frames": self.config.max_frames,
                "max_images_per_post": self.config.max_images_per_post,
                "max_dynamic_media_per_post": self.config.max_dynamic_media_per_post,
                "video_decoder": "PyAV bounded deterministic frame sampling",
            },
            "local_snapshot": snapshot_identity,
            "packages": {
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils"),
            },
            "device_type": self.device.type,
        }
