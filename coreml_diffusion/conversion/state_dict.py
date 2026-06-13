"""State-dict layout predicates — framework-free (no coremltools/diffusers).

Single-file checkpoints come in two layouts: original LDM (UNet keys under
``model.diffusion_model.``) and diffusers-native UNet-only dumps (block keys at
the top level — e.g. ``LCM_Dreamshaper_v7_4k.safetensors``, the canonical
full-distill LCM artifact). diffusers' ``from_single_file`` only understands
the former and raises ``SingleFileComponentError`` on the latter;
``convert.load_unet`` routes on this predicate.
"""

DIFFUSERS_UNET_KEY_PREFIXES = ("down_blocks.", "up_blocks.", "mid_block.")
LDM_UNET_KEY_PREFIX = "model.diffusion_model."


def is_diffusers_unet_layout(keys) -> bool:
    """True when ``keys`` form a diffusers-format UNet-only state dict."""
    keys = list(keys)
    has_diffusers_blocks = any(k.startswith(DIFFUSERS_UNET_KEY_PREFIXES) for k in keys)
    has_ldm_prefix = any(k.startswith(LDM_UNET_KEY_PREFIX) for k in keys)
    return has_diffusers_blocks and not has_ldm_prefix
