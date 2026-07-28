"""Masked post mean pooling and an MLP head over frozen Qwen3-VL embeddings."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ocd_v3.experiments.qwen_vl_mean_mlp_config import QwenVLMeanMLPModelConfig


class QwenVLMeanMLPClassifier(nn.Module):
    def __init__(self, *, embedding_dimension: int, config: QwenVLMeanMLPModelConfig) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self.embedding_dimension = embedding_dimension
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dimension, 1),
        )

    def forward(
        self,
        content_embeddings: torch.Tensor,
        metadata: torch.Tensor,
        post_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if content_embeddings.ndim != 4 or content_embeddings.shape[2] != 1:
            raise ValueError(
                "Qwen3-VL mean pooling expects [batch, post, one representation, dim]"
            )
        if content_embeddings.shape[3] != self.embedding_dimension:
            raise ValueError("Content embedding dimension differs from the configured input")
        if post_mask.shape != content_embeddings.shape[:2]:
            raise ValueError("Post mask shape differs from the content sequence")
        if metadata.shape != (*content_embeddings.shape[:2], 0):
            raise ValueError("Qwen3-VL mean-MLP baseline excludes metadata")
        values = content_embeddings[:, :, 0, :]
        float_mask = post_mask.unsqueeze(-1).to(values.dtype)
        user_embedding = (values * float_mask).sum(dim=1) / float_mask.sum(
            dim=1
        ).clamp_min(1.0)
        logits = self.classifier(user_embedding).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "user_embedding": user_embedding,
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
