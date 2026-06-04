"""Tier 1 smoke: convert a tiny AutoencoderKL's decoder and encoder through the
real ``convert.convert_vae_*`` path and confirm the .mlpackages load with the I/O
we declare.

Uses a genuine (but micro) ``diffusers.AutoencoderKL`` so the production code runs
end to end — including the mid-block self-attention routed through the ORIGINAL
processor and the post/quant conv submodule wiring — without needing a real SD
checkpoint or the ANE. Runs in seconds on a macOS-ARM runner.

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""

import platform
import shutil

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)

# latent 8x8 -> image 64x64; tiny channels so conversion finishes in seconds.
LATENT_SIZE = (8, 8)
LATENT_CHANNELS = 4


@pytest.fixture(scope="module")
def tiny_vae():
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D"],
        block_out_channels=[32],
        latent_channels=LATENT_CHANNELS,
        layers_per_block=1,
        sample_size=64,
    ).eval()


def _io_names(pkg_path):
    import coremltools as ct

    spec = ct.models.MLModel(str(pkg_path)).get_spec()
    return (
        {i.name for i in spec.description.input},
        {o.name for o in spec.description.output},
    )


def test_vae_decoder_converts_with_declared_io(tiny_vae, tmp_path):
    from coreml_diffusion.convert import convert_vae_decoder

    pkg = tmp_path / "vae_decoder.mlpackage"
    convert_vae_decoder(tiny_vae, str(pkg), batch_size=1, sample_size=LATENT_SIZE)

    inputs, outputs = _io_names(pkg)
    assert inputs == {"latent"}
    assert "image" in outputs
    shutil.rmtree(pkg, ignore_errors=True)


def test_vae_encoder_converts_with_declared_io(tiny_vae, tmp_path):
    from coreml_diffusion.convert import convert_vae_encoder

    pkg = tmp_path / "vae_encoder.mlpackage"
    convert_vae_encoder(tiny_vae, str(pkg), batch_size=1, sample_size=LATENT_SIZE)

    inputs, outputs = _io_names(pkg)
    assert inputs == {"image"}
    assert "latent_moments" in outputs
    shutil.rmtree(pkg, ignore_errors=True)
