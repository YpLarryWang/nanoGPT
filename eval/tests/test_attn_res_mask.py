from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "hf_nanogpt"))

import modeling_nanogpt as hf_model  # noqa: E402


class IdentityNorm(torch.nn.Module):
    def forward(self, value):
        return value


class AttnResMaskTest(unittest.TestCase):
    def test_none_mix_is_exact_legacy_formula(self):
        torch.manual_seed(7)
        sources = [torch.randn(2, 3, 8) for _ in range(4)]
        query = torch.randn(8)
        norm = IdentityNorm()
        values = torch.stack(sources)
        logits = torch.einsum("c,sbtc->sbt", query, norm(values))
        expected = torch.einsum(
            "sbt,sbtc->btc", logits.softmax(dim=0), values
        )
        actual = hf_model.attn_res_mix(sources, query, norm)
        self.assertTrue(torch.equal(actual, expected))

    def test_old_mask_is_block_age_aware(self):
        # block-start q1: partial is age 1, so every non-embedding block is old.
        start_q1 = hf_model.build_attn_res_keep_mask(
            num_blocks=4,
            layer_idx=4,
            router_site="q1",
            block_start=True,
            mode="old",
            control_seed=1,
        )
        self.assertEqual(start_q1.tolist(), [True, False, False, False, True])

        # Inside the block (and at q2), the last item in blocks is age 1.
        inside_q1 = hf_model.build_attn_res_keep_mask(
            num_blocks=4,
            layer_idx=5,
            router_site="q1",
            block_start=False,
            mode="old",
            control_seed=1,
        )
        inside_q2 = hf_model.build_attn_res_keep_mask(
            num_blocks=4,
            layer_idx=4,
            router_site="q2",
            block_start=True,
            mode="old",
            control_seed=1,
        )
        expected = [True, False, False, True, True]
        self.assertEqual(inside_q1.tolist(), expected)
        self.assertEqual(inside_q2.tolist(), expected)

    def test_embed_and_count_matched_controls(self):
        embed = hf_model.build_attn_res_keep_mask(
            num_blocks=3,
            layer_idx=3,
            router_site="q2",
            block_start=False,
            mode="embed",
            control_seed=11,
        )
        self.assertEqual(embed.tolist(), [False, True, True, True])

        kwargs = dict(
            num_blocks=6,
            layer_idx=7,
            router_site="q2",
            block_start=False,
            control_seed=1337,
        )
        old = hf_model.build_attn_res_keep_mask(mode="old", **kwargs)
        random_a = hf_model.build_attn_res_keep_mask(
            mode="random_count_matched", **kwargs
        )
        random_b = hf_model.build_attn_res_keep_mask(
            mode="random_count_matched", **kwargs
        )
        self.assertTrue(torch.equal(random_a, random_b))
        self.assertEqual(int((~random_a).sum()), int((~old).sum()))
        self.assertTrue(bool(random_a[0]))
        self.assertTrue(bool(random_a[-1]))

    def test_five_control_seeds_preserve_counts_and_change_some_sites(self):
        signatures = []
        for seed in (20260718, 20260719, 20260720, 20260721, 20260722):
            site_masks = []
            for layer_idx, router_site, block_start in (
                (7, "q1", False),
                (12, "q2", True),
                (19, "q1", True),
            ):
                kwargs = dict(
                    num_blocks=10,
                    layer_idx=layer_idx,
                    router_site=router_site,
                    block_start=block_start,
                    control_seed=seed,
                )
                old = hf_model.build_attn_res_keep_mask(mode="old", **kwargs)
                random = hf_model.build_attn_res_keep_mask(
                    mode="random_count_matched", **kwargs
                )
                self.assertEqual(int((~random).sum()), int((~old).sum()))
                site_masks.append(tuple(random.tolist()))
            signatures.append(tuple(site_masks))
        self.assertGreater(len(set(signatures)), 1)

    def test_masked_mix_zeroes_forbidden_sources_and_renormalizes(self):
        sources = [torch.full((1, 1, 2), float(i)) for i in range(4)]
        keep = torch.tensor([True, False, True, False])
        mixed = hf_model.attn_res_mix(
            sources, torch.zeros(2), IdentityNorm(), keep
        )
        self.assertTrue(torch.equal(mixed, torch.full((1, 1, 2), 1.0)))

    def test_only_q1_q2_are_masked_not_final_router(self):
        original_mode = hf_model._ATTN_RES_MASK_MODE
        original_mix = hf_model.attn_res_mix
        seen = []

        def traced_mix(sources, query, norm, keep_mask=None):
            seen.append(None if keep_mask is None else keep_mask.clone())
            return original_mix(sources, query, norm, keep_mask)

        config = hf_model.NanoGPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=4,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            use_rmsnorm=True,
            use_swiglu=False,
            use_rope=True,
            use_attn_res=True,
            attn_res_block_size=2,
        )
        try:
            hf_model._ATTN_RES_MASK_MODE = "old"
            hf_model.attn_res_mix = traced_mix
            model = hf_model.NanoGPTForCausalLM(config).eval()
            with torch.no_grad():
                model(input_ids=torch.randint(0, config.vocab_size, (1, 5)))
        finally:
            hf_model.attn_res_mix = original_mix
            hf_model._ATTN_RES_MASK_MODE = original_mode

        self.assertEqual(len(seen), 2 * config.n_layer + 1)
        self.assertIsNone(seen[-1])
        self.assertTrue(any(mask is not None and bool((~mask).any()) for mask in seen[:-1]))


if __name__ == "__main__":
    unittest.main()
