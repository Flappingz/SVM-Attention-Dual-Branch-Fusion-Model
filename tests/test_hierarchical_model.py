from __future__ import annotations

import importlib.util
import unittest

from ocd_v3.experiments.full_config import HierarchicalModelConfig


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class HierarchicalModelTests(unittest.TestCase):
    def test_architecture_ablations_remove_only_declared_attention_levels(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        common = dict(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="none",
            ffn_hidden_dim=16,
            classifier_hidden_dim=16,
            dropout=0.0,
            user_pooling="mean",
            attention_inner_dim=4,
        )
        no_post = HierarchicalAttentionClassifier(
            config=HierarchicalModelConfig(
                **common, architecture_variant="without_post_cross_attention"
            ),
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        )
        no_user = HierarchicalAttentionClassifier(
            config=HierarchicalModelConfig(
                **common, architecture_variant="without_user_self_attention"
            ),
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        )
        both_mean = HierarchicalAttentionClassifier(
            config=HierarchicalModelConfig(
                **common, architecture_variant="mean_pooling_at_both_levels"
            ),
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        ).eval()

        self.assertFalse(
            any("post_encoder.attention" in name for name, _ in no_post.named_parameters())
        )
        self.assertTrue(
            any("user_encoder.attention" in name for name, _ in no_post.named_parameters())
        )
        self.assertTrue(
            any("post_encoder.attention" in name for name, _ in no_user.named_parameters())
        )
        self.assertFalse(
            any("user_encoder.attention" in name for name, _ in no_user.named_parameters())
        )
        self.assertFalse(
            any("encoder.attention" in name for name, _ in both_mean.named_parameters())
        )
        self.assertFalse(
            any("encoder.feed_forward" in name for name, _ in both_mean.named_parameters())
        )

        content = torch.randn(1, 4, 1, 16)
        metadata = torch.randn(1, 4, 5)
        mask = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.inference_mode():
            original = both_mean(content, metadata, mask)["logits"]
            permuted = both_mean(
                content[:, permutation], metadata[:, permutation], mask
            )["logits"]
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))

    def test_linear_head_preserves_backbone_and_reduces_classifier_parameters(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        common = dict(
            d_model=256,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=512,
            classifier_hidden_dim=512,
            dropout=0.5,
            user_pooling="mean",
        )
        mlp = HierarchicalAttentionClassifier(
            config=HierarchicalModelConfig(**common),
            representation_names=["final"],
            embedding_dimension=2048,
            metadata_dimension=5,
        )
        linear = HierarchicalAttentionClassifier(
            config=HierarchicalModelConfig(**common, classifier_head="linear"),
            representation_names=["final"],
            embedding_dimension=2048,
            metadata_dimension=5,
        )

        mlp_counts = mlp.trainable_parameter_counts()
        linear_counts = linear.trainable_parameter_counts()
        self.assertEqual(mlp_counts["classifier_head"], 132_609)
        self.assertEqual(linear_counts["classifier_head"], 769)
        self.assertEqual(mlp_counts["non_classifier"], linear_counts["non_classifier"])
        self.assertEqual(mlp_counts["total"], 1_189_131)
        self.assertEqual(linear_counts["total"], 1_057_291)
        self.assertEqual(len(mlp.classifier), 5)
        self.assertEqual(len(linear.classifier), 2)
        self.assertIsInstance(linear.classifier[0], torch.nn.LayerNorm)
        self.assertEqual(tuple(linear.classifier[1].weight.shape), (1, 256))

    def test_compact_full_256_preserves_all_blocks_and_reduces_internal_widths(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=256,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=128,
            classifier_hidden_dim=128,
            dropout=0.5,
            user_pooling="mean",
            attention_inner_dim=64,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=2048,
            metadata_dimension=5,
        )

        self.assertIsInstance(model.post_encoder.attention, torch.nn.Module)
        self.assertIsInstance(model.post_encoder.feed_forward, torch.nn.Module)
        self.assertIsInstance(model.user_encoder.attention, torch.nn.Module)
        self.assertIsInstance(model.user_encoder.feed_forward, torch.nn.Module)
        self.assertEqual(
            tuple(model.post_encoder.attention.query_projection.weight.shape),
            (64, 256),
        )
        self.assertEqual(
            tuple(model.post_encoder.attention.output_projection.weight.shape),
            (256, 64),
        )
        self.assertEqual(
            tuple(model.post_encoder.feed_forward.layers[0].weight.shape),
            (128, 256),
        )
        self.assertEqual(
            tuple(model.user_encoder.feed_forward.layers[3].weight.shape),
            (256, 128),
        )
        self.assertEqual(tuple(model.classifier[1].weight.shape), (128, 256))
        self.assertEqual(tuple(model.classifier[4].weight.shape), (1, 128))
        self.assertEqual(
            model.trainable_parameter_breakdown(),
            {
                "content_projection": 0,
                "metadata_projection": 1_546,
                "post_cross_attention_and_ffn": 133_696,
                "user_self_attention_and_ffn": 132_928,
                "terminal_mlp": 33_537,
                "total": 301_707,
            },
        )
        self.assertEqual(model.trainable_parameter_count(), 301_707)

    def test_raw_qwen_mean_skip_fuses_full_embedding_with_hierarchical_path(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=256,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="none",
            ffn_hidden_dim=128,
            classifier_hidden_dim=128,
            dropout=0.0,
            user_pooling="mean",
            attention_inner_dim=64,
            raw_mean_skip_dim=128,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=2048,
            metadata_dimension=5,
        )
        content = torch.randn(2, 4, 1, 2048)
        metadata = torch.randn(2, 4, 5)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        outputs = model(content, metadata, mask)

        self.assertEqual(tuple(outputs["user_embedding"].shape), (2, 256))
        self.assertEqual(tuple(outputs["classifier_input"].shape), (2, 384))
        self.assertEqual(tuple(model.raw_mean_skip[1].weight.shape), (128, 2048))
        self.assertEqual(tuple(model.classifier[1].weight.shape), (128, 384))
        self.assertEqual(
            model.trainable_parameter_breakdown(),
            {
                "content_projection": 0,
                "metadata_projection": 1_546,
                "post_cross_attention_and_ffn": 133_696,
                "user_self_attention_and_ffn": 132_928,
                "terminal_mlp": 50_177,
                "raw_qwen_mean_skip": 266_368,
                "total": 584_715,
            },
        )
        self.assertEqual(model.trainable_parameter_count(), 584_715)

    def test_raw_qwen_mean_skip_requires_final_representation(self) -> None:
        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="none",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
            raw_mean_skip_dim=4,
        )
        with self.assertRaisesRegex(ValueError, "final Qwen"):
            HierarchicalAttentionClassifier(
                config=config,
                representation_names=["layer:1"],
                embedding_dimension=16,
                metadata_dimension=5,
            )

    def test_preaggregated_raw_mean_matches_full_width_forward(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="none",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
            attention_inner_dim=4,
            raw_mean_skip_dim=4,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=0,
        ).eval()
        content = torch.randn(2, 4, 1, 16)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        metadata = torch.empty((2, 4, 0))
        float_mask = mask.unsqueeze(-1).to(content.dtype)
        raw_mean = (content[:, :, 0, :] * float_mask).sum(1) / float_mask.sum(1)
        with torch.inference_mode():
            full = model(content, metadata, mask)["logits"]
            compact = model(
                content[..., :8],
                metadata,
                mask,
                raw_user_mean=raw_mean,
            )["logits"]
        self.assertTrue(torch.allclose(full, compact, atol=1e-6, rtol=1e-6))

    def test_attention_mask_is_safe_in_float16(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import ExplicitMultiHeadAttention

        attention = ExplicitMultiHeadAttention(4, 2, 0.0).half()
        values = torch.ones((1, 2, 4), dtype=torch.float16)
        output, weights = attention(
            values,
            values,
            values,
            key_mask=torch.tensor([[True, False]]),
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(float(weights[..., 1].sum().detach()), 0.0)

    def test_forward_uses_both_attention_levels_and_masks_padding(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=4,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final", "layer:1"],
            embedding_dimension=8,
            metadata_dimension=5,
        )
        content = torch.randn(2, 4, 2, 8)
        metadata = torch.randn(2, 4, 5)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        outputs = model(content, metadata, mask, return_attention=True)
        self.assertEqual(tuple(outputs["logits"].shape), (2,))
        self.assertEqual(tuple(outputs["post_attention"].shape[:2]), (2, 4))
        self.assertEqual(tuple(outputs["post_attention"].shape[-2:]), (1, 3))
        self.assertEqual(tuple(outputs["user_attention"].shape[-2:]), (4, 4))
        self.assertFalse(any(name == "threshold" for name, _ in model.named_parameters()))
        self.assertFalse(
            any("latent_queries" in name for name, _ in model.named_parameters())
        )

    def test_position_encoding_makes_post_order_observable(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        torch.manual_seed(7)
        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=16,
            classifier_hidden_dim=16,
            dropout=0.0,
            user_pooling="mean",
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        ).eval()
        content = torch.randn(1, 4, 1, 16)
        metadata = torch.randn(1, 4, 5)
        mask = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([3, 1, 0, 2])
        with torch.inference_mode():
            original = model(content, metadata, mask)["user_embedding"]
            permuted = model(
                content[:, permutation], metadata[:, permutation], mask
            )["user_embedding"]
        self.assertFalse(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))

    def test_explicit_no_position_encoder_is_identity(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="none",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
            attention_inner_dim=4,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        )
        values = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        expected = values * mask.unsqueeze(-1)
        self.assertTrue(torch.equal(model.user_position_encoding(values, mask), expected))

    def test_relative_position_bias_preserves_embeddings_and_encodes_order(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        torch.manual_seed(11)
        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="relative_bias",
            ffn_hidden_dim=16,
            classifier_hidden_dim=16,
            dropout=0.0,
            user_pooling="mean",
            relative_position_max_distance=3,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=16,
            metadata_dimension=5,
        ).eval()
        values = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        self.assertTrue(torch.equal(model.user_position_encoding(values, mask), values))

        relative_bias = model.user_encoder.relative_position_bias
        self.assertIsNotNone(relative_bias)
        assert relative_bias is not None
        with torch.no_grad():
            relative_bias.bias.weight.copy_(
                torch.arange(14, dtype=torch.float32).reshape(7, 2)
            )
        content = torch.randn(1, 4, 1, 16)
        metadata = torch.randn(1, 4, 5)
        permutation = torch.tensor([3, 1, 0, 2])
        with torch.inference_mode():
            original = model(content, metadata, mask)["user_embedding"]
            permuted = model(
                content[:, permutation], metadata[:, permutation], mask
            )["user_embedding"]
        self.assertFalse(torch.allclose(original, permuted, atol=1e-6, rtol=1e-6))

    def test_text_only_head_excludes_metadata_and_learns_input_projection(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=8,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=16,
            classifier_hidden_dim=16,
            dropout=0.0,
            user_pooling="mean",
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["text"],
            embedding_dimension=12,
            metadata_dimension=0,
        )
        content = torch.randn(2, 4, 1, 12)
        metadata = torch.empty(2, 4, 0)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        outputs = model(content, metadata, mask, return_attention=True)
        self.assertEqual(tuple(outputs["logits"].shape), (2,))
        self.assertEqual(tuple(outputs["post_attention"].shape[-2:]), (1, 1))
        names = {name for name, _ in model.named_parameters()}
        self.assertIn("representation_projections.0.weight", names)
        self.assertFalse(any(name.startswith("metadata_projection") for name in names))

    def test_final_embedding_uses_512_prefix_then_projects_to_256(self) -> None:
        import torch

        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=256,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=512,
            classifier_hidden_dim=512,
            dropout=0.0,
            user_pooling="mean",
            embedding_prefix_dim=512,
        )
        model = HierarchicalAttentionClassifier(
            config=config,
            representation_names=["final"],
            embedding_dimension=2048,
            metadata_dimension=0,
        )
        projection = model.representation_projections[0]
        self.assertIsInstance(projection, torch.nn.Linear)
        self.assertEqual(tuple(projection.weight.shape), (256, 512))

        content = torch.randn(2, 3, 1, 2048)
        changed_tail = content.clone()
        changed_tail[..., 512:] += 1000.0
        with torch.inference_mode():
            first = model._project_content(content)
            second = model._project_content(changed_tail)
        self.assertEqual(tuple(first.shape), (2, 3, 1, 256))
        self.assertTrue(torch.equal(first, second))

    def test_final_embedding_rejects_prefix_wider_than_feature(self) -> None:
        from ocd_v3.modeling.hierarchical import HierarchicalAttentionClassifier

        config = HierarchicalModelConfig(
            d_model=4,
            post_attention_heads=2,
            user_attention_heads=2,
            user_position_encoding="sinusoidal",
            ffn_hidden_dim=8,
            classifier_hidden_dim=8,
            dropout=0.0,
            user_pooling="mean",
            embedding_prefix_dim=9,
        )
        with self.assertRaisesRegex(ValueError, "embedding_prefix_dim"):
            HierarchicalAttentionClassifier(
                config=config,
                representation_names=["final"],
                embedding_dimension=8,
                metadata_dimension=0,
            )


if __name__ == "__main__":
    unittest.main()
