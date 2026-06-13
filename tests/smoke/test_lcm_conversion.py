"""Tier 1 smoke: the generic LCM conversion path on a synthetic micro-UNet.

Locks the E-LCM fixes: the ``timestep_cond`` trace input takes its dim from
``config.time_cond_proj_dim`` (not a hardcoded 256), an LCM-LoRA merge (no
guidance embedding) is rejected with a clear error, and diffusers-layout
UNet-only dumps (the canonical full-distill LCM artifact shape) load through
``load_unet_from_diffusers_state_dict``.

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""

import platform

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)

GUIDANCE_DIM = 16  # deliberately != 256 to prove the dim comes from the config

TINY_ARCH = dict(
    sample_size=8,
    layers_per_block=1,
    block_out_channels=(32, 64),
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    cross_attention_dim=32,
    attention_head_dim=8,
    norm_num_groups=8,
)


def _tiny_unet(time_cond_proj_dim=None):
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(**TINY_ARCH, time_cond_proj_dim=time_cond_proj_dim)


def test_lcm_unet_converts_with_config_guidance_dim(tmp_path):
    import coremltools as ct

    from coreml_diffusion.convert import convert_unet
    from coreml_diffusion.model_version import ModelVersion

    torch.manual_seed(0)
    out_path = str(tmp_path / "lcm_unet.mlpackage")
    convert_unet(
        _tiny_unet(time_cond_proj_dim=GUIDANCE_DIM),
        ModelVersion.LCM,
        out_path,
        sample_size=(8, 8),
    )

    spec = ct.models.MLModel(out_path, skip_model_load=True).get_spec()
    inputs = {
        i.name: tuple(i.type.multiArrayType.shape) for i in spec.description.input
    }
    assert "timestep_cond" in inputs
    assert inputs["timestep_cond"] == (1, GUIDANCE_DIM)


def test_lcm_merge_checkpoint_is_rejected(tmp_path):
    from coreml_diffusion.convert import convert_unet
    from coreml_diffusion.model_version import ModelVersion

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="LCM-LoRA merge"):
        convert_unet(
            _tiny_unet(time_cond_proj_dim=None),
            ModelVersion.LCM,
            str(tmp_path / "merge.mlpackage"),
            sample_size=(8, 8),
        )


def test_diffusers_layout_dump_loads_with_guidance_embedding(tmp_path):
    from safetensors.torch import save_file

    from coreml_diffusion.convert import load_unet_from_diffusers_state_dict

    torch.manual_seed(0)
    source = _tiny_unet(time_cond_proj_dim=GUIDANCE_DIM)
    dump = tmp_path / "tiny_lcm_dump.safetensors"
    save_file(source.state_dict(), str(dump))

    # cross_attention_dim and time_cond_proj_dim must be read from the weights;
    # only the miniature architecture is supplied as overrides.
    loaded = load_unet_from_diffusers_state_dict(
        str(dump), **{k: v for k, v in TINY_ARCH.items() if k != "cross_attention_dim"}
    )

    assert loaded.config.cross_attention_dim == TINY_ARCH["cross_attention_dim"]
    assert loaded.config.time_cond_proj_dim == GUIDANCE_DIM
    assert torch.equal(
        loaded.time_embedding.cond_proj.weight, source.time_embedding.cond_proj.weight
    )


def test_lcm_inputs_legacy_call_keeps_256():
    # The Suite's LCM converter calls lcm_inputs without a ref UNet; the legacy
    # 256 fallback is part of the import contract until the Suite migrates.
    from coreml_diffusion.convert import lcm_inputs

    inputs = lcm_inputs({"sample": torch.rand(2, 4, 8, 8)})
    assert inputs["timestep_cond"].shape == (2, 256)
