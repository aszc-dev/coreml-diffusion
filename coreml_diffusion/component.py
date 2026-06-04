"""Convertible model components — framework-free leaf.

A single-file checkpoint bundles several sub-models; ``coreml_diffusion`` can
convert each into its own ``.mlpackage``. This enum is the canonical identifier
set, mirrored into the discovery contract (``list_convertible_components``) and
the naming contract (``compose_component_name``).

``UNET`` keeps its historical conversion path and filename (``compose_out_name``);
the other components are the additive VAE / text-encoder extension. ``.value`` is
the wire string used by the CLI ``--component`` flag and the discovery list — kept
lowercase and stable, since a saved workflow / benchmark manifest references it
verbatim (same additive-only rule as ``ModelVersion``).

``TEXT_ENCODER_2`` is SDXL-only (its second CLIP, ``CLIPTextModelWithProjection``);
on SD1.5 only ``TEXT_ENCODER`` exists. Validity per model version is enforced at
convert time, not encoded here.
"""

from enum import Enum


class Component(Enum):
    UNET = "unet"
    VAE_DECODER = "vae_decoder"
    VAE_ENCODER = "vae_encoder"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"


# Declaration order is the discovery-list order; UNET leads to match the
# historical, primary conversion path.
CONVERTIBLE_COMPONENTS = tuple(c.value for c in Component)
