"""Tier 1 smoke: the ORIGINAL attention path is convertible and fp16-safe.

Two regressions are guarded here:

1. ORIGINAL must not be a no-op. It used to ``return unet`` unchanged, leaving
   diffusers' stock ``AttnProcessor2_0`` (SDPA), which does not even convert under
   coremltools 9 (its ``view(B, -1, heads, d)`` reshapes fail the torch frontend).
   ``apply_attention_implementation(unet, "ORIGINAL")`` must install our
   conversion-safe ``OriginalAttnProcessor`` instead.

2. The ORIGINAL kernel upcasts QK^T / softmax to fp32. Run in fp16, full
   self-attention at SD1.5's highest-resolution block overflows fp16 in the score
   matmul -> inf -> NaN; the fp32 score path keeps it finite.

Needs only torch + diffusers (no coremltools), but lives in Tier 1 because
tests/unit/ must stay free of heavy imports.
"""

import platform

import pytest
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from coreml_diffusion.conversion.attention import (
    OriginalAttnProcessor,
    apply_attention_implementation,
    original,
)

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)


def _tiny_unet():
    """Minimal UNet2DConditionModel with cross-attention blocks (so it owns a few
    ``Attention`` modules) — small enough to construct in milliseconds."""
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def test_original_installs_convertible_processor():
    """The bug was that ORIGINAL returned the UNet unchanged, leaving the stock
    SDPA processor (which does not convert under ct9). Lock in: the default is
    SDPA, and ORIGINAL swaps every Attention to OriginalAttnProcessor."""
    torch.manual_seed(0)
    unet = _tiny_unet()
    attn_modules = [m for m in unet.modules() if isinstance(m, Attention)]
    assert attn_modules, "expected the tiny UNet to own Attention modules"

    # Pre-condition: diffusers' default is SDPA.
    assert all(isinstance(m.processor, AttnProcessor2_0) for m in attn_modules)

    apply_attention_implementation(unet, "ORIGINAL")

    # Post-condition: our conversion-safe full-attention processor everywhere.
    assert all(isinstance(m.processor, OriginalAttnProcessor) for m in attn_modules)


def test_original_upcast_keeps_fp16_attention_finite():
    """Reproduce the failure mode: a large QK^T overflows fp16 to NaN, but the
    ORIGINAL kernel's fp32 score path keeps the output finite."""
    torch.manual_seed(0)
    heads, dim_head, seq = 8, 40, 64
    channels = heads * dim_head

    # [B, C, 1, S] layout (what _attention_forward hands the kernel). Large q/k so
    # the score matmul exceeds fp16's 65504 ceiling.
    q = (torch.randn(1, channels, 1, seq) * 60).half()
    k = (torch.randn(1, channels, 1, seq) * 60).half()
    v = torch.randn(1, channels, 1, seq).half()

    # Naive fp16 scores (no upcast) overflow -> inf -> NaN after softmax.
    mh_q = q.view(1, heads, dim_head, -1)
    mh_k = k.view(1, heads, dim_head, -1)
    naive_scores = torch.einsum("becq,beck->bkeq", mh_q, mh_k) * (dim_head**-0.5)
    assert torch.isnan(naive_scores.softmax(dim=1)).any(), "expected fp16 overflow"

    # original() upcasts the score matmul + softmax to fp32 -> finite output.
    out = original(q, k, v, None, heads, dim_head)
    assert torch.isfinite(out).all()
    assert out.shape == (1, channels, 1, seq)
