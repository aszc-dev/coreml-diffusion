"""Tier 0: model-version auto-detection from checkpoint weights.

Locks the architecture fingerprinting that lets ``convert(model_version=None)``
pick the right conversion path: cross-attention context dim (attn2.to_k) plus
the presence of a guidance embedding (cond_proj). Synthetic safetensors files
carry only the two keys the detector reads, so the test stays framework-free.
"""

import pytest
import torch
from safetensors.torch import save_file

from coreml_diffusion.conversion.state_dict import detect_model_version
from coreml_diffusion.model_version import ModelVersion


def _write_ckpt(path, cross_attention_dim, *, guidance=False, with_cross_attn=True):
    tensors = {}
    if with_cross_attn:
        # attn2.to_k maps the context dim -> inner dim; shape[1] is what we read.
        key = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
        tensors[key] = torch.zeros(320, cross_attention_dim)
    if guidance:
        tensors["time_embedding.cond_proj.weight"] = torch.zeros(320, 256)
    if not tensors:
        tensors["dummy"] = torch.zeros(1)
    save_file(tensors, str(path))
    return str(path)


@pytest.mark.parametrize(
    "cross_attention_dim, guidance, expected",
    [
        (768, False, ModelVersion.SD15),
        (768, True, ModelVersion.LCM),
        (2048, False, ModelVersion.SDXL),
        (1280, False, ModelVersion.SDXL_REFINER),
    ],
)
def test_detects_known_architectures(tmp_path, cross_attention_dim, guidance, expected):
    ckpt = _write_ckpt(
        tmp_path / "m.safetensors", cross_attention_dim, guidance=guidance
    )
    assert detect_model_version(ckpt) is expected


def test_lcm_lora_merge_detects_as_sd15(tmp_path):
    # An LCM-LoRA merge is plain SD1.5 architecture (no guidance embedding); it
    # must NOT be mistaken for a full-distill LCM.
    ckpt = _write_ckpt(tmp_path / "merge.safetensors", 768, guidance=False)
    assert detect_model_version(ckpt) is ModelVersion.SD15


def test_guidance_with_non_sd15_dim_is_rejected(tmp_path):
    ckpt = _write_ckpt(tmp_path / "sdxl_lcm.safetensors", 2048, guidance=True)
    with pytest.raises(ValueError, match="only SD1.5-class LCM"):
        detect_model_version(ckpt)


def test_unknown_cross_attention_dim_is_rejected(tmp_path):
    # 1024 is SD2.x — unsupported; the error must name the observed dim.
    ckpt = _write_ckpt(tmp_path / "sd2.safetensors", 1024)
    with pytest.raises(ValueError, match="cross_attention_dim=1024"):
        detect_model_version(ckpt)


def test_no_cross_attention_weights_is_rejected(tmp_path):
    ckpt = _write_ckpt(tmp_path / "weird.safetensors", 768, with_cross_attn=False)
    with pytest.raises(ValueError, match="no cross-attention"):
        detect_model_version(ckpt)


def test_non_safetensors_is_rejected(tmp_path):
    bogus = tmp_path / "model.ckpt"
    bogus.write_bytes(b"not safetensors")
    with pytest.raises(ValueError, match="not a readable safetensors"):
        detect_model_version(str(bogus))
