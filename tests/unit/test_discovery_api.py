"""Gate E1: freeze the coreml_diffusion discovery contract.

These tests pin the discovery functions to the *exact* strings the ComfyUI node
renders today, so wiring the node's INPUT_TYPES onto them (E3) cannot change any
dropdown value or break a saved workflow. They also prove the package imports in a
framework-free env (Tier-0): if `import coreml_diffusion` pulled comfy/coremltools/
diffusers, test_tier0_purity would fail the whole `-m unit` run.

The literals below are duplicated from the node's current hardcodes on purpose —
this file is the contract anchor, so it must fail loudly if a source list drifts.
"""

import coreml_diffusion
from coreml_diffusion.attention import ATTENTION_IMPLEMENTATIONS
from coreml_diffusion.model_version import ModelVersion
from coreml_diffusion.naming import QUANT_NBITS_VALUES

# ---------- model versions ---------------------------------------------------


def test_list_model_versions_verified_only_matches_node_dropdown():
    # nodes.py renders [ModelVersion.SD15.name, ModelVersion.SDXL.name]
    assert coreml_diffusion.list_model_versions() == ["SD15", "SDXL"]


def test_list_model_versions_experimental_adds_lcm_and_refiner():
    versions = coreml_diffusion.list_model_versions(include_experimental=True)
    assert set(versions) == {"SD15", "SDXL", "SDXL_REFINER", "LCM"}
    # VERIFIED ones still lead, in declaration order.
    assert versions[:2] == ["SD15", "SDXL"]


def test_model_versions_emit_name_reversible_by_node():
    # The node reverses the dropdown string via ModelVersion[...] (name lookup).
    # Emitting .value ("sd15") would KeyError; this guards against that regression.
    for name in coreml_diffusion.list_model_versions(include_experimental=True):
        assert ModelVersion[name]  # raises KeyError if a .value leaked in


# ---------- attention impls --------------------------------------------------


def test_list_attention_impls_matches_node_dropdown():
    assert coreml_diffusion.list_attention_impls() == [
        "SPLIT_EINSUM",
        "SPLIT_EINSUM_V2",
        "ORIGINAL",
    ]


def test_list_attention_impls_tracks_source_tuple():
    assert coreml_diffusion.list_attention_impls() == list(ATTENTION_IMPLEMENTATIONS)


# ---------- quant modes ------------------------------------------------------


def test_list_quant_modes_matches_node_dropdown():
    assert coreml_diffusion.list_quant_modes() == ["none", "8", "6", "4"]


def test_list_quant_modes_tracks_source_tuple():
    assert coreml_diffusion.list_quant_modes() == list(QUANT_NBITS_VALUES)


# ---------- convertible components -------------------------------------------


def test_list_convertible_components_exact():
    assert coreml_diffusion.list_convertible_components() == [
        "unet",
        "vae_decoder",
        "vae_encoder",
        "text_encoder",
        "text_encoder_2",
    ]


def test_list_convertible_components_tracks_source_tuple():
    from coreml_diffusion.component import CONVERTIBLE_COMPONENTS

    assert coreml_diffusion.list_convertible_components() == list(
        CONVERTIBLE_COMPONENTS
    )


def test_unet_leads_components_for_backcompat():
    # "unet" stays first: the historical default conversion path.
    assert coreml_diffusion.list_convertible_components()[0] == "unet"


# ---------- contract version -------------------------------------------------


def test_contract_version_is_a_string():
    assert isinstance(coreml_diffusion.CONTRACT_VERSION, str)
    # 1.1: list_convertible_components added (additive minor bump).
    assert coreml_diffusion.CONTRACT_VERSION == "1.1"
