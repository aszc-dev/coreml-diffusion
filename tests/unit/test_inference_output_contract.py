"""Tier-0 contract for the CoreML text-encoder output wrapper.

``_CoreMLTextEncoderOutput`` is pure torch (no coremltools/diffusers), so it is
exercisable in the framework-free tier. It pins the exact access pattern the
diffusers pipelines rely on: ``out[0]`` (pooled when present, else embeds) and
``out.hidden_states[-2]`` (the penultimate state SDXL concatenates).
"""

import pytest
import torch

from coreml_diffusion.inference import _CoreMLTextEncoderOutput


def test_index_zero_is_embeds_when_no_pooled():
    # SD1.5 path: no pooled vector -> [0] is the (final) hidden state.
    embeds = torch.randn(1, 77, 768)
    out = _CoreMLTextEncoderOutput(embeds, None)
    assert torch.equal(out[0], embeds)
    assert torch.equal(out.last_hidden_state, embeds)


def test_index_zero_is_pooled_when_present():
    # SDXL encoder 2: [0] is the projected pooled vector.
    embeds = torch.randn(1, 77, 1280)
    pooled = torch.randn(1, 1280)
    out = _CoreMLTextEncoderOutput(embeds, pooled)
    assert torch.equal(out[0], pooled)
    assert torch.equal(out.text_embeds, pooled)
    assert torch.equal(out.pooler_output, pooled)


def test_penultimate_hidden_state_is_the_baked_embeds():
    # SDXL reads hidden_states[-2]; the converter baked exactly that tensor.
    embeds = torch.randn(1, 77, 768)
    out = _CoreMLTextEncoderOutput(embeds, None)
    assert torch.equal(out.hidden_states[-2], embeds)


def test_only_index_zero_is_exposed():
    out = _CoreMLTextEncoderOutput(torch.randn(1, 77, 768), None)
    with pytest.raises(IndexError):
        out[1]
