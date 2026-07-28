from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ocd_v3.experiments.full_config import HierarchicalModelConfig

_EPSILON = 1e-8


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class ExplicitMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        inner_dim: int | None = None,
    ) -> None:
        super().__init__()
        resolved_inner_dim = d_model if inner_dim is None else inner_dim
        if resolved_inner_dim < 1 or resolved_inner_dim % num_heads:
            raise ValueError("attention inner_dim must be positive and divisible by num_heads")
        self.d_model = d_model
        self.inner_dim = resolved_inner_dim
        self.num_heads = num_heads
        self.head_dim = resolved_inner_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = nn.Linear(d_model, resolved_inner_dim)
        self.key_projection = nn.Linear(d_model, resolved_inner_dim)
        self.value_projection = nn.Linear(d_model, resolved_inner_dim)
        self.output_projection = nn.Linear(resolved_inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        queries = self.query_projection(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = self.key_projection(key).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.value_projection(value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            if attention_bias.ndim == 3:
                attention_bias = attention_bias.unsqueeze(0)
            expected = (batch_size, self.num_heads, query_length, key_length)
            if attention_bias.ndim != 4 or any(
                actual not in {1, target}
                for actual, target in zip(attention_bias.shape, expected, strict=True)
            ):
                raise ValueError(
                    "attention_bias must broadcast to [batch, head, query, key]"
                )
            scores = scores + attention_bias.to(device=scores.device, dtype=scores.dtype)
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )
        weights = torch.softmax(scores, dim=-1)
        if key_mask is not None:
            weights = weights * key_mask[:, None, None, :].to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(_EPSILON)
        if query_mask is not None:
            weights = weights * query_mask[:, None, :, None].to(weights.dtype)
        weights = self.dropout(weights)
        context = torch.matmul(weights, values)
        if query_mask is not None:
            context = context * query_mask[:, None, :, None].to(context.dtype)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.inner_dim
        )
        output = self.output_projection(context)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output, weights


class PostCrossAttentionEncoder(nn.Module):
    def __init__(self, config: HierarchicalModelConfig) -> None:
        super().__init__()
        self.class_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.context_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.attention = ExplicitMultiHeadAttention(
            config.d_model,
            config.post_attention_heads,
            config.dropout,
            inner_dim=config.attention_inner_dim,
        )
        self.feed_forward = FeedForward(
            config.d_model, config.ffn_hidden_dim, config.dropout
        )

    def forward(
        self, post_tokens: torch.Tensor, post_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, post_count, token_count, d_model = post_tokens.shape
        flattened = post_tokens.reshape(batch_size * post_count, token_count, d_model)
        queries = self.class_query.expand(batch_size * post_count, -1, -1)
        query_mask = post_mask.reshape(batch_size * post_count, 1).expand(
            -1, queries.shape[1]
        )
        attended, weights = self.attention(
            self.query_norm(queries),
            self.context_norm(flattened),
            self.context_norm(flattened),
            query_mask=query_mask,
        )
        encoded = queries + attended
        encoded = encoded + self.feed_forward(self.ffn_norm(encoded))
        post_embeddings = encoded[:, 0, :].reshape(batch_size, post_count, d_model)
        post_embeddings = post_embeddings * post_mask.unsqueeze(-1).to(post_embeddings.dtype)
        return post_embeddings, weights.reshape(
            batch_size,
            post_count,
            weights.shape[1],
            weights.shape[2],
            weights.shape[3],
        )


class PostMeanEncoder(nn.Module):
    """Replace post-level cross-attention with an arithmetic token mean.

    The single-level ablation retains the post FFN so that the controlled
    difference is the cross-attention operation.  The two-level mean-pooling
    ablation removes both attention/FFN blocks.
    """

    def __init__(self, config: HierarchicalModelConfig, *, retain_ffn: bool) -> None:
        super().__init__()
        self.ffn_norm = nn.LayerNorm(config.d_model) if retain_ffn else None
        self.feed_forward = (
            FeedForward(config.d_model, config.ffn_hidden_dim, config.dropout)
            if retain_ffn
            else None
        )

    def forward(
        self, post_tokens: torch.Tensor, post_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = post_tokens.shape[2]
        encoded = post_tokens.mean(dim=2)
        if self.feed_forward is not None and self.ffn_norm is not None:
            encoded = encoded + self.feed_forward(self.ffn_norm(encoded))
        encoded = encoded * post_mask.unsqueeze(-1).to(encoded.dtype)
        weights = post_mask[:, :, None, None, None].to(encoded.dtype).expand(
            -1, -1, 1, 1, token_count
        ) / float(token_count)
        return encoded, weights


class SinusoidalPositionEncoding(nn.Module):
    """Add fixed absolute post-rank encodings without introducing new parameters."""

    def __init__(self, d_model: int, base: float = 10_000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.d_model:
            raise ValueError("Position encoding expects [batch, post, d_model]")
        if mask.shape != values.shape[:2]:
            raise ValueError("Position mask shape differs from post sequence")
        length = values.shape[1]
        positions = torch.arange(length, device=values.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=values.device, dtype=torch.float32)
            * (-math.log(self.base) / self.d_model)
        )
        encoding = torch.zeros(
            length, self.d_model, device=values.device, dtype=torch.float32
        )
        angles = positions * frequencies
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        encoding = encoding.to(values.dtype).unsqueeze(0)
        return (values + encoding) * mask.unsqueeze(-1).to(values.dtype)


class IdentityPositionEncoding(nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values * mask.unsqueeze(-1).to(values.dtype)


class LearnedRelativePositionBias(nn.Module):
    """Add signed relative post-rank information to attention logits only."""

    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        if num_heads < 1 or max_distance < 1:
            raise ValueError("Relative position bias dimensions must be positive")
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance + 1, num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(
        self,
        query_length: int,
        key_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_length, device=device).unsqueeze(1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        relative_positions = (key_positions - query_positions).clamp(
            -self.max_distance, self.max_distance
        )
        indices = relative_positions + self.max_distance
        return self.bias(indices).permute(2, 0, 1).to(dtype=dtype)


class UserSelfAttentionEncoder(nn.Module):
    def __init__(self, config: HierarchicalModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.attention = ExplicitMultiHeadAttention(
            config.d_model,
            config.user_attention_heads,
            config.dropout,
            inner_dim=config.attention_inner_dim,
        )
        self.relative_position_bias = (
            LearnedRelativePositionBias(
                config.user_attention_heads,
                config.relative_position_max_distance,
            )
            if config.user_position_encoding == "relative_bias"
            and config.relative_position_max_distance is not None
            else None
        )
        self.feed_forward = FeedForward(
            config.d_model, config.ffn_hidden_dim, config.dropout
        )

    def forward(
        self, post_embeddings: torch.Tensor, post_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(post_embeddings)
        attention_bias = (
            self.relative_position_bias(
                normalized.shape[1],
                normalized.shape[1],
                device=normalized.device,
                dtype=normalized.dtype,
            )
            if self.relative_position_bias is not None
            else None
        )
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            query_mask=post_mask,
            key_mask=post_mask,
            attention_bias=attention_bias,
        )
        encoded = post_embeddings + attended
        encoded = encoded + self.feed_forward(self.ffn_norm(encoded))
        encoded = encoded * post_mask.unsqueeze(-1).to(encoded.dtype)
        return encoded, weights


class UserFeedForwardEncoder(nn.Module):
    """Remove user self-attention while retaining its token-wise FFN."""

    def __init__(self, config: HierarchicalModelConfig) -> None:
        super().__init__()
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(
            config.d_model, config.ffn_hidden_dim, config.dropout
        )

    def forward(
        self, post_embeddings: torch.Tensor, post_mask: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        encoded = post_embeddings + self.feed_forward(self.ffn_norm(post_embeddings))
        encoded = encoded * post_mask.unsqueeze(-1).to(encoded.dtype)
        return encoded, None


class IdentityUserEncoder(nn.Module):
    """Pass projected post means directly to user-level mean pooling."""

    def forward(
        self, post_embeddings: torch.Tensor, post_mask: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return post_embeddings * post_mask.unsqueeze(-1).to(post_embeddings.dtype), None


class HierarchicalAttentionClassifier(nn.Module):
    """Paper-faithful post cross-attention followed by user self-attention."""

    def __init__(
        self,
        *,
        config: HierarchicalModelConfig,
        representation_names: Sequence[str],
        embedding_dimension: int,
        metadata_dimension: int,
    ) -> None:
        super().__init__()
        if not representation_names:
            raise ValueError("At least one content representation is required")
        self.embedding_prefix_dim = config.embedding_prefix_dim or config.d_model
        if (
            embedding_dimension < self.embedding_prefix_dim
            and "final" in representation_names
        ):
            raise ValueError("Final MRL embedding is narrower than embedding_prefix_dim")
        self.config = config
        self.representation_names = tuple(representation_names)
        if config.raw_mean_skip_dim is not None and "final" not in self.representation_names:
            raise ValueError("raw_mean_skip_dim requires the final Qwen representation")
        self.representation_projections = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if self.embedding_prefix_dim == config.d_model
                    else nn.Linear(self.embedding_prefix_dim, config.d_model)
                )
                if name == "final"
                else nn.Linear(embedding_dimension, config.d_model)
                for name in self.representation_names
            ]
        )
        if metadata_dimension < 0:
            raise ValueError("metadata_dimension cannot be negative")
        self.metadata_dimension = metadata_dimension
        self.metadata_projection = (
            nn.Sequential(
                nn.LayerNorm(metadata_dimension),
                nn.Linear(metadata_dimension, config.d_model),
            )
            if metadata_dimension > 0
            else None
        )
        if config.architecture_variant in {
            "without_post_cross_attention",
            "mean_pooling_at_both_levels",
        }:
            self.post_encoder = PostMeanEncoder(
                config,
                retain_ffn=config.architecture_variant == "without_post_cross_attention",
            )
        else:
            self.post_encoder = PostCrossAttentionEncoder(config)
        self.user_position_encoding = (
            SinusoidalPositionEncoding(config.d_model)
            if config.user_position_encoding == "sinusoidal"
            and config.architecture_variant != "mean_pooling_at_both_levels"
            else IdentityPositionEncoding()
        )
        if config.architecture_variant == "without_user_self_attention":
            self.user_encoder = UserFeedForwardEncoder(config)
        elif config.architecture_variant == "mean_pooling_at_both_levels":
            self.user_encoder = IdentityUserEncoder()
        else:
            self.user_encoder = UserSelfAttentionEncoder(config)
        self.raw_mean_skip = (
            nn.Sequential(
                nn.LayerNorm(embedding_dimension),
                nn.Linear(embedding_dimension, config.raw_mean_skip_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.raw_mean_skip_dim is not None
            else None
        )
        classifier_input_dim = config.d_model + (config.raw_mean_skip_dim or 0)
        self.classifier = (
            nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Linear(classifier_input_dim, config.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.classifier_hidden_dim, 1),
            )
            if config.classifier_head == "mlp"
            else nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Linear(classifier_input_dim, 1),
            )
        )

    def _project_content(self, content: torch.Tensor) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for index, (name, projection) in enumerate(
            zip(self.representation_names, self.representation_projections, strict=True)
        ):
            values = content[:, :, index, :]
            projected.append(
                projection(values[:, :, : self.embedding_prefix_dim])
                if name == "final"
                else projection(values)
            )
        return torch.stack(projected, dim=2)

    def forward(
        self,
        content_embeddings: torch.Tensor,
        metadata: torch.Tensor,
        post_mask: torch.Tensor,
        *,
        raw_user_mean: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        if content_embeddings.ndim != 4:
            raise ValueError(
                "content_embeddings must have shape [batch, post, representation, dim]"
            )
        projected_content = self._project_content(content_embeddings)
        if self.metadata_projection is None:
            if metadata.shape != (*content_embeddings.shape[:2], 0):
                raise ValueError(
                    "Text-only hierarchical input requires zero-width metadata"
                )
            post_tokens = projected_content
        else:
            if metadata.shape != (
                *content_embeddings.shape[:2],
                self.metadata_dimension,
            ):
                raise ValueError("Metadata shape differs from the configured schema")
            metadata_token = self.metadata_projection(metadata).unsqueeze(2)
            post_tokens = torch.cat([projected_content, metadata_token], dim=2)
        post_embeddings, post_attention = self.post_encoder(post_tokens, post_mask)
        positioned_posts = self.user_position_encoding(post_embeddings, post_mask)
        encoded_posts, user_attention = self.user_encoder(positioned_posts, post_mask)
        float_mask = post_mask.unsqueeze(-1).to(encoded_posts.dtype)
        user_embedding = (encoded_posts * float_mask).sum(dim=1) / float_mask.sum(
            dim=1
        ).clamp_min(1.0)
        classifier_input = user_embedding
        if self.raw_mean_skip is not None:
            if raw_user_mean is None:
                final_index = self.representation_names.index("final")
                raw_posts = content_embeddings[:, :, final_index, :]
                raw_user_mean = (raw_posts * float_mask).sum(dim=1) / float_mask.sum(
                    dim=1
                ).clamp_min(1.0)
            elif raw_user_mean.shape != (
                content_embeddings.shape[0],
                self.raw_mean_skip[0].normalized_shape[0],
            ):
                raise ValueError("Pre-aggregated raw_user_mean has an unexpected shape")
            classifier_input = torch.cat(
                [user_embedding, self.raw_mean_skip(raw_user_mean)], dim=-1
            )
        logits = self.classifier(classifier_input).squeeze(-1)
        result: dict[str, Any] = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "user_embedding": user_embedding,
            "classifier_input": classifier_input,
        }
        if return_attention:
            result["post_attention"] = post_attention
            if user_attention is not None:
                result["user_attention"] = user_attention
        return result

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_counts(self) -> dict[str, int]:
        classifier = sum(
            parameter.numel()
            for parameter in self.classifier.parameters()
            if parameter.requires_grad
        )
        total = self.trainable_parameter_count()
        return {
            "classifier_head": classifier,
            "non_classifier": total - classifier,
            "total": total,
        }

    def trainable_parameter_breakdown(self) -> dict[str, int]:
        """Report the paper-defined downstream classifier by architectural block."""

        def count(module: nn.Module | None) -> int:
            return (
                sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
                if module is not None
                else 0
            )

        values = {
            "content_projection": count(self.representation_projections),
            "metadata_projection": count(self.metadata_projection),
            "terminal_mlp": count(self.classifier),
        }
        post_name = (
            "post_cross_attention_and_ffn"
            if self.config.architecture_variant == "full"
            or self.config.architecture_variant == "without_user_self_attention"
            else "post_mean_and_ffn"
            if self.config.architecture_variant == "without_post_cross_attention"
            else "post_mean_pooling"
        )
        user_name = (
            "user_self_attention_and_ffn"
            if self.config.architecture_variant == "full"
            or self.config.architecture_variant == "without_post_cross_attention"
            else "user_ffn_without_self_attention"
            if self.config.architecture_variant == "without_user_self_attention"
            else "user_mean_pooling"
        )
        values[post_name] = count(self.post_encoder)
        values[user_name] = count(self.user_encoder)
        if self.raw_mean_skip is not None:
            values["raw_qwen_mean_skip"] = count(self.raw_mean_skip)
        values["total"] = sum(values.values())
        return values
