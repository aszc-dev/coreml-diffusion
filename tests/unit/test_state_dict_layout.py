"""Tier 0: state-dict layout detection routing ``load_unet``.

Locks the predicate that decides whether a single-file checkpoint is an LDM
checkpoint (``from_single_file`` path) or a diffusers-layout UNet-only dump
(direct state-dict load — the canonical full-distill LCM artifact shape).
"""

from coreml_diffusion.conversion.state_dict import is_diffusers_unet_layout

DIFFUSERS_UNET_KEYS = [
    "conv_in.weight",
    "time_embedding.linear_1.weight",
    "time_embedding.cond_proj.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "mid_block.resnets.0.conv1.weight",
    "up_blocks.3.resnets.2.conv2.weight",
]

LDM_CHECKPOINT_KEYS = [
    "model.diffusion_model.time_embed.0.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
    "first_stage_model.decoder.conv_in.weight",
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
]


def test_diffusers_unet_dump_is_detected():
    assert is_diffusers_unet_layout(DIFFUSERS_UNET_KEYS)


def test_ldm_checkpoint_is_not_diffusers_layout():
    assert not is_diffusers_unet_layout(LDM_CHECKPOINT_KEYS)


def test_mixed_prefixes_resolve_to_ldm():
    # An LDM file whose extras coincidentally include diffusers-looking keys
    # must still go through from_single_file.
    assert not is_diffusers_unet_layout(LDM_CHECKPOINT_KEYS + DIFFUSERS_UNET_KEYS)


def test_unrelated_keys_are_not_diffusers_layout():
    assert not is_diffusers_unet_layout(["text_model.encoder.layers.0.mlp.fc1.weight"])
    assert not is_diffusers_unet_layout([])
