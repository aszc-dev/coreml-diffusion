"""Characterization tests for non-UNet component .mlpackage filenames.

``compose_component_name`` is the cache key for VAE / text-encoder artifacts, the
sibling of ``compose_out_name`` (which stays UNet-only). Locks the stem rules and
the resolution-dependence split: VAE names carry WxH, text-encoder names do not.
"""

import pytest

from coreml_diffusion.naming import compose_component_name

# ---------- VAE (resolution-dependent) --------------------------------------


@pytest.mark.parametrize("component", ["vae_decoder", "vae_encoder"])
def test_vae_includes_resolution(component):
    out = compose_component_name(
        ckpt_name="dreamshaper_8.safetensors",
        component=component,
        batch_size=1,
        width=512,
        height=512,
    )
    assert out == f"dreamshaper_8_{component}_1x512x512"


def test_vae_decoder_batch_and_size():
    out = compose_component_name(
        ckpt_name="dreamshaper_8.safetensors",
        component="vae_decoder",
        batch_size=2,
        width=768,
        height=1024,
    )
    assert out == "dreamshaper_8_vae_decoder_2x768x1024"


# ---------- text encoder (resolution-independent) ---------------------------


@pytest.mark.parametrize("component", ["text_encoder", "text_encoder_2"])
def test_text_encoder_omits_resolution(component):
    out = compose_component_name(
        ckpt_name="sd_xl_base_1.0.safetensors",
        component=component,
        batch_size=2,
        width=1024,
        height=1024,
    )
    # resolution is intentionally absent — the artifact is resolution-independent.
    assert out == f"sd_xl_base_1_{component}_2"


# ---------- stem massage (mirrors compose_out_name) -------------------------


def test_drops_extension_at_first_period():
    out = compose_component_name(
        ckpt_name="my.checkpoint.v2.safetensors",
        component="vae_decoder",
        batch_size=1,
        width=512,
        height=512,
    )
    assert out == "my_vae_decoder_1x512x512"


def test_replaces_spaces_with_underscores():
    out = compose_component_name(
        ckpt_name="dream shaper 8.safetensors",
        component="text_encoder",
        batch_size=1,
        width=512,
        height=512,
    )
    assert out == "dream_shaper_8_text_encoder_1"


# ---------- quantization ----------------------------------------------------


def test_quant_none_appends_nothing():
    out = compose_component_name(
        ckpt_name="dreamshaper_8.safetensors",
        component="vae_decoder",
        batch_size=1,
        width=512,
        height=512,
        quantize_nbits="none",
    )
    assert out == "dreamshaper_8_vae_decoder_1x512x512"


@pytest.mark.parametrize("nbits,suffix", [("4", "_q4"), ("6", "_q6"), ("8", "_q8")])
def test_quant_appends_q_suffix(nbits, suffix):
    out = compose_component_name(
        ckpt_name="dreamshaper_8.safetensors",
        component="vae_decoder",
        batch_size=1,
        width=512,
        height=512,
        quantize_nbits=nbits,
    )
    assert out == f"dreamshaper_8_vae_decoder_1x512x512{suffix}"


def test_text_encoder_quant_suffix():
    out = compose_component_name(
        ckpt_name="sd_xl_base_1.0.safetensors",
        component="text_encoder_2",
        batch_size=1,
        width=1024,
        height=1024,
        quantize_nbits="6",
    )
    assert out == "sd_xl_base_1_text_encoder_2_1_q6"


# ---------- guards ----------------------------------------------------------


def test_unet_component_rejected():
    # "unet" owns compose_out_name; routing it here is a programming error.
    with pytest.raises(ValueError, match="unet"):
        compose_component_name(
            ckpt_name="x.safetensors",
            component="unet",
            batch_size=1,
            width=512,
            height=512,
        )


def test_unknown_component_rejected():
    with pytest.raises(ValueError, match="component"):
        compose_component_name(
            ckpt_name="x.safetensors",
            component="refiner",
            batch_size=1,
            width=512,
            height=512,
        )


def test_invalid_quant_rejected():
    with pytest.raises(ValueError, match="quantize_nbits"):
        compose_component_name(
            ckpt_name="x.safetensors",
            component="vae_decoder",
            batch_size=1,
            width=512,
            height=512,
            quantize_nbits="16",
        )
