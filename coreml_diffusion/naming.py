"""Pure out_name composition for the Core ML UNet artifact.

Extracted from CoreMLConverter.convert so the filename contract
can be tested + reused without instantiating the node. The string is the
cache key: every workflow that references a converted .mlpackage depends
on it staying byte-for-byte identical.
"""

from typing import Iterable, Tuple

ATTN_SUFFIX = {
    "SPLIT_EINSUM": "se",
    "SPLIT_EINSUM_V2": "se2",
    "ORIGINAL": "orig",
}

# Palettization bits. "none" = no quantization (default; keeps the
# unquantized filename intact so existing workflows still resolve their
# cached .mlpackage). Numeric values append a `_q<bits>` suffix.
QUANT_NBITS_VALUES = ("none", "8", "6", "4")

# Filename infix per non-UNet component. UNET is absent on purpose: it keeps the
# historical stem composed by ``compose_out_name`` and must never route here.
# VAE artifacts depend on resolution (the latent/image spatial dims are baked at
# trace time); text-encoder artifacts do not (only token count, fixed at 77, and
# batch matter), so their stems omit WxH to avoid redundant per-resolution copies.
COMPONENT_NAME_SUFFIX = {
    "vae_decoder": "vae_decoder",
    "vae_encoder": "vae_encoder",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}

_RESOLUTION_DEPENDENT_COMPONENTS = {"vae_decoder", "vae_encoder"}


def compose_out_name(
    *,
    ckpt_name: str,
    batch_size: int,
    width: int,
    height: int,
    controlnet_support: bool,
    attention_implementation: str,
    lora_names: Iterable[str] = (),
    quantize_nbits: str = "none",
) -> str:
    """Build the .mlpackage stem from convert() parameters.

    Locked behaviour (characterization tests):
      - first '.' in ckpt_name wins (`a.b.c.safetensors` -> `a`)
      - spaces collapse to underscores
      - LoRA names are taken stem-only, sorted, joined with '_' and
        prefixed with '_' when present (caller is expected to pass a
        sorted list; we sort defensively)
      - controlnet adds `_cn`
      - attn suffix is `_se` | `_se2` | `_orig`

    Quantization:
      - quantize_nbits "none" (default) appends nothing — existing
        unquantized .mlpackages keep the old filename
      - "4" / "6" / "8" appends `_q<bits>` after the attn suffix
    """
    if quantize_nbits not in QUANT_NBITS_VALUES:
        raise ValueError(
            f"quantize_nbits={quantize_nbits!r} not in {QUANT_NBITS_VALUES}"
        )
    stem = ckpt_name.split(".")[0]
    sorted_names = sorted(lora_names)
    lora_str = (
        "_" + "_".join(name.split(".")[0] for name in sorted_names)
        if sorted_names
        else ""
    )
    cn_suffix = "_cn" if controlnet_support else ""
    attn_suffix = "_" + ATTN_SUFFIX[attention_implementation]
    quant_suffix = f"_q{quantize_nbits}" if quantize_nbits != "none" else ""
    out_name = (
        f"{stem}{lora_str}_{batch_size}x{width}x{height}"
        f"{cn_suffix}{attn_suffix}{quant_suffix}"
    )
    return out_name.replace(" ", "_")


def compose_component_name(
    *,
    ckpt_name: str,
    component: str,
    batch_size: int,
    width: int,
    height: int,
    quantize_nbits: str = "none",
) -> str:
    """Build the ``.mlpackage`` stem for a non-UNet component (cache key).

    Mirrors ``compose_out_name``'s stem rules (first '.' wins, spaces -> '_') but
    omits the UNet-only axes (attention impl, LoRA, ControlNet). VAE components
    carry the resolution; text-encoder components do not (resolution-independent).

      vae_decoder / vae_encoder : ``{stem}_{component}_{batch}x{width}x{height}{quant}``
      text_encoder / _2         : ``{stem}_{component}_{batch}{quant}``

    Quantization mirrors ``compose_out_name``: "none" appends nothing.
    ``component`` must be a non-UNet key of ``COMPONENT_NAME_SUFFIX``; "unet"
    raises (it owns ``compose_out_name``).
    """
    if component not in COMPONENT_NAME_SUFFIX:
        raise ValueError(
            f"component={component!r} not in {tuple(COMPONENT_NAME_SUFFIX)}; "
            "'unet' uses compose_out_name"
        )
    if quantize_nbits not in QUANT_NBITS_VALUES:
        raise ValueError(
            f"quantize_nbits={quantize_nbits!r} not in {QUANT_NBITS_VALUES}"
        )
    stem = ckpt_name.split(".")[0]
    suffix = COMPONENT_NAME_SUFFIX[component]
    quant_suffix = f"_q{quantize_nbits}" if quantize_nbits != "none" else ""
    if component in _RESOLUTION_DEPENDENT_COMPONENTS:
        size = f"_{batch_size}x{width}x{height}"
    else:
        size = f"_{batch_size}"
    out_name = f"{stem}_{suffix}{size}{quant_suffix}"
    return out_name.replace(" ", "_")


def lora_names_from_params(lora_params: Iterable[Tuple[str, float]]) -> list[str]:
    """Mirror the sort applied inside CoreMLConverter.convert."""
    return [name for name, _ in sorted(lora_params, key=lambda pair: pair[0])]
