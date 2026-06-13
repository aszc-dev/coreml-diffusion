"""State-dict layout predicates — framework-free (no coremltools/diffusers).

Single-file checkpoints come in two layouts: original LDM (UNet keys under
``model.diffusion_model.``) and diffusers-native UNet-only dumps (block keys at
the top level — e.g. ``LCM_Dreamshaper_v7_4k.safetensors``, the canonical
full-distill LCM artifact). diffusers' ``from_single_file`` only understands
the former and raises ``SingleFileComponentError`` on the latter;
``convert.load_unet`` routes on this predicate.

Also home to ``detect_model_version``: it reads only the safetensors header (no
coremltools/diffusers), so the conversion entrypoint can auto-pick the model
version without dragging the heavy stack into the discovery path.
"""

from coreml_diffusion.model_version import ModelVersion

DIFFUSERS_UNET_KEY_PREFIXES = ("down_blocks.", "up_blocks.", "mid_block.")
LDM_UNET_KEY_PREFIX = "model.diffusion_model."

# cross_attention_dim -> model version. The context (key/value) dim of the UNet's
# cross-attention is the architecture fingerprint: SD1.5 conditions on a single
# 768-dim CLIP, SDXL on the 2048-dim concat of both encoders, the SDXL refiner on
# the 1280-dim OpenCLIP-bigG alone. A guidance embedding (time_embedding.cond_proj)
# on top of the 768-dim SD1.5 stack marks a full-distill LCM.
_CROSS_ATTENTION_DIM_TO_VERSION = {
    768: ModelVersion.SD15,
    2048: ModelVersion.SDXL,
    1280: ModelVersion.SDXL_REFINER,
}


def is_diffusers_unet_layout(keys) -> bool:
    """True when ``keys`` form a diffusers-format UNet-only state dict."""
    keys = list(keys)
    has_diffusers_blocks = any(k.startswith(DIFFUSERS_UNET_KEY_PREFIXES) for k in keys)
    has_ldm_prefix = any(k.startswith(LDM_UNET_KEY_PREFIX) for k in keys)
    return has_diffusers_blocks and not has_ldm_prefix


def safetensors_keys(ckpt_path):
    """The file's key list when it is safetensors, else None.

    Probes by content, not filename — a resolved checkpoint path may point at
    an extension-less blob (e.g. inside the Hugging Face cache).
    """
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(ckpt_path, framework="pt") as f:
            return list(f.keys())
    except SafetensorError:
        return None


def detect_model_version(ckpt_path):
    """Infer the ``ModelVersion`` from a checkpoint's UNet weights.

    Reads two architecture fingerprints straight from the safetensors header
    (no full model load): the cross-attention context dim (``attn2.to_k``) and
    whether a guidance embedding (``cond_proj``) is present. Works for both LDM
    and diffusers key layouts. Raises ``ValueError`` carrying the observed
    evidence when the architecture is unrecognised, so a bad guess is debuggable
    rather than silent.
    """
    keys = safetensors_keys(ckpt_path)
    if keys is None:
        raise ValueError(
            f"Cannot auto-detect model version from {ckpt_path!r}: not a readable "
            "safetensors file. Pass model_version explicitly."
        )
    cross_attn_key = next((k for k in keys if k.endswith("attn2.to_k.weight")), None)
    if cross_attn_key is None:
        raise ValueError(
            f"Cannot auto-detect model version from {ckpt_path!r}: no cross-attention "
            "(attn2.to_k) weights found. Pass model_version explicitly."
        )
    from safetensors import safe_open

    with safe_open(ckpt_path, framework="pt") as f:
        cross_attention_dim = f.get_slice(cross_attn_key).get_shape()[1]
    has_guidance_embedding = any(k.endswith("cond_proj.weight") for k in keys)

    if has_guidance_embedding:
        if cross_attention_dim == 768:
            return ModelVersion.LCM
        raise ValueError(
            f"Cannot auto-detect model version from {ckpt_path!r}: found a guidance "
            f"embedding (LCM) but cross_attention_dim={cross_attention_dim}; only "
            "SD1.5-class LCM (cross_attention_dim=768) is supported. Pass "
            "model_version explicitly."
        )
    version = _CROSS_ATTENTION_DIM_TO_VERSION.get(cross_attention_dim)
    if version is None:
        raise ValueError(
            f"Cannot auto-detect model version from {ckpt_path!r}: unrecognised "
            f"cross_attention_dim={cross_attention_dim}. Supported: 768 (SD15/LCM), "
            "2048 (SDXL), 1280 (SDXL_REFINER). Pass model_version explicitly."
        )
    return version
