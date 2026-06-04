"""Tier 1 smoke: round-trip tiny converted packages through the inference adapters.

Proves the e2e plumbing end to end on CPU — convert a micro VAE / CLIP, load the
saved ``.mlpackage`` back through ``CoreMLVAE`` / ``CoreMLTextEncoder``, run a real
``predict``, and check the diffusers-facing contract (``DecoderOutput.sample``,
``latent_dist``, ``out[0]`` / ``hidden_states[-2]``). No real checkpoint or ANE;
the full-image golden is the Tier 2 anchor.

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""

import platform

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)

LATENT_SIZE = (8, 8)
SEQ_LEN = 77
HIDDEN = 32
PROJECTION = 16


def _tiny_vae():
    """A micro AutoencoderKL with the real 8x scale factor (4 blocks) so the
    latent<->image spatial relationship the converter assumes holds."""
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[4, 4, 8, 8],
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=2,
        sample_size=64,
    ).eval()


def _tiny_clip_config():
    from transformers import CLIPTextConfig

    return CLIPTextConfig(
        vocab_size=1000,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=SEQ_LEN,
        projection_dim=PROJECTION,
    )


def _convert_text_encoder(encoder, output_names, pkg, *, index, pooled):
    import coremltools as ct

    from coreml_diffusion.conversion.text_encoder import (
        CoreMLTextEncoderWrapper,
        static_causal_mask,
    )
    from coreml_diffusion.convert import convert_to_coreml

    wrapper = CoreMLTextEncoderWrapper(
        encoder.eval(), hidden_states_index=index, output_pooled=pooled
    ).eval()
    ids_shape = (1, SEQ_LEN)
    with static_causal_mask(SEQ_LEN):
        traced = torch.jit.trace(wrapper, torch.zeros(ids_shape, dtype=torch.int64))
    inputs = [ct.TensorType(name="input_ids", shape=ids_shape, dtype=np.int32)]
    model = convert_to_coreml("te", traced, inputs, output_names, str(pkg))
    model.save(str(pkg))


def test_coreml_vae_decode_and_encode(tmp_path):
    from coreml_diffusion.convert import convert_vae_decoder, convert_vae_encoder
    from coreml_diffusion.inference import CoreMLVAE

    vae = _tiny_vae()
    dec = tmp_path / "vae_decoder.mlpackage"
    enc = tmp_path / "vae_encoder.mlpackage"
    convert_vae_decoder(vae, str(dec), batch_size=1, sample_size=LATENT_SIZE)
    convert_vae_encoder(vae, str(enc), batch_size=1, sample_size=LATENT_SIZE)

    coreml_vae = CoreMLVAE(
        vae,
        decoder_mlpackage=str(dec),
        encoder_mlpackage=str(enc),
        compute_unit="CPU_ONLY",
    )

    image = coreml_vae.decode(torch.rand(1, 4, *LATENT_SIZE), return_dict=False)[0]
    assert image.shape == (1, 3, LATENT_SIZE[0] * 8, LATENT_SIZE[1] * 8)

    posterior = coreml_vae.encode(torch.rand(1, 3, 64, 64)).latent_dist
    assert posterior.mean.shape == (1, 4, *LATENT_SIZE)


def test_coreml_text_encoder_sd15_shape(tmp_path):
    from transformers import CLIPTextModel

    from coreml_diffusion.inference import CoreMLTextEncoder

    torch.manual_seed(0)
    encoder = CLIPTextModel(_tiny_clip_config())
    pkg = tmp_path / "text_encoder.mlpackage"
    _convert_text_encoder(encoder, ["hidden_states"], pkg, index=None, pooled=False)

    coreml_te = CoreMLTextEncoder(str(pkg), encoder, compute_unit="CPU_ONLY")
    out = coreml_te(torch.zeros(1, SEQ_LEN, dtype=torch.long))

    assert out[0].shape == (1, SEQ_LEN, HIDDEN)  # no pooled -> [0] is embeds
    assert out.hidden_states[-2].shape == (1, SEQ_LEN, HIDDEN)


def test_coreml_text_encoder_sdxl_pooled(tmp_path):
    from transformers import CLIPTextModelWithProjection

    from coreml_diffusion.inference import CoreMLTextEncoder

    torch.manual_seed(0)
    encoder = CLIPTextModelWithProjection(_tiny_clip_config())
    pkg = tmp_path / "text_encoder_2.mlpackage"
    _convert_text_encoder(
        encoder, ["hidden_states", "pooled_embeds"], pkg, index=-2, pooled=True
    )

    coreml_te = CoreMLTextEncoder(str(pkg), encoder, compute_unit="CPU_ONLY")
    out = coreml_te(torch.zeros(1, SEQ_LEN, dtype=torch.long))

    assert out[0].shape == (1, PROJECTION)  # pooled present -> [0] is pooled
    assert out.hidden_states[-2].shape == (1, SEQ_LEN, HIDDEN)
