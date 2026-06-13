"""Core ML UNet conversion mechanics — framework-free.

Moved from ``coreml_suite/converter.py`` in extraction phase E2. This module
produces a ``.mlpackage`` on disk and stops there: it must NOT import ``comfy``,
``folder_paths``, or ``comfy_extras``. Output paths are inputs, not resolved here.

``get_sample_input`` carries an optional ``scheduler`` so the LCM path (which
derives the trace timestep from an LCM scheduler) shares this single
implementation instead of keeping a near-duplicate copy.
"""

import gc
import os
import time

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel

from coreml_diffusion.attention import ATTENTION_IMPLEMENTATIONS
from coreml_diffusion.component import Component
from coreml_diffusion.conversion.attention import apply_attention_implementation
from coreml_diffusion.conversion.shapes import conv2d_output_shape
from coreml_diffusion.conversion.state_dict import is_diffusers_unet_layout
from coreml_diffusion.conversion.text_encoder import (
    CoreMLTextEncoderWrapper,
    static_causal_mask,
)
from coreml_diffusion.conversion.trace import prepare_unet_for_coreml_trace
from coreml_diffusion.conversion.unet import CoreMLUNetWrapper
from coreml_diffusion.conversion.vae import (
    CoreMLVAEDecoderWrapper,
    CoreMLVAEEncoderWrapper,
)
from coreml_diffusion.logger import logger
from coreml_diffusion.model_version import ModelVersion

DEFAULT_TRACE_TIMESTEP = 999.0
TEXT_TOKEN_SEQUENCE_LENGTH = 77


def get_unet(model_version: ModelVersion, ref_unet, attention_implementation):
    ref_unet = prepare_unet_for_coreml_trace(ref_unet)
    unet = apply_attention_implementation(
        ref_unet.eval(),
        attention_implementation,
    )
    # The freshly built wrapper defaults to training mode; the inner UNet is
    # already eval, but coremltools inspects the top-level traced module and warns
    # ("Model is not in eval mode"). eval() on the wrapper silences it and makes
    # the eval-mode trace explicit (output is unchanged — UNet dropout p=0).
    return CoreMLUNetWrapper(unet, model_version).eval()


def get_encoder_hidden_states_shape(ref_unet, batch_size):
    encoder_hidden_states_shape = (
        batch_size,
        TEXT_TOKEN_SEQUENCE_LENGTH,
        ref_unet.config.cross_attention_dim,
    )

    return encoder_hidden_states_shape


def get_coreml_inputs(sample_inputs):
    coreml_sample_unet_inputs = {
        k: v.numpy().astype(np.float16) for k, v in sample_inputs.items()
    }
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        )
        for k, v in coreml_sample_unet_inputs.items()
    ]


def load_coreml_model(out_path):
    logger.info(f"Loading model from {out_path}")

    start = time.time()
    coreml_model = ct.models.MLModel(out_path)
    logger.info(f"Loading {out_path} took {time.time() - start:.1f} seconds")

    return coreml_model


def convert_to_coreml(
    submodule_name, torchscript_module, sample_inputs, output_names, out_path
):
    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        coreml_model = load_coreml_model(out_path)
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=sample_inputs,
            outputs=[
                ct.TensorType(name=name, dtype=np.float32) for name in output_names
            ],
            skip_model_load=True,
        )

        del torchscript_module
        gc.collect()

    return coreml_model


def get_sample_input(
    batch_size, encoder_hidden_states_shape, sample_shape, scheduler=None
):
    """Build the example inputs used to JIT-trace the UNet.

    When ``scheduler`` is provided (the LCM path) the trace timestep is taken
    from ``scheduler.timesteps[0]``; otherwise the fixed ``DEFAULT_TRACE_TIMESTEP``
    is used. Only the shapes/dtypes/order of these tensors matter to the traced
    graph — the random values are placeholders.
    """
    timestep_value = (
        scheduler.timesteps[0].item()
        if scheduler is not None
        else DEFAULT_TRACE_TIMESTEP
    )
    sample_unet_inputs = dict(
        [
            ("sample", torch.rand(*sample_shape)),
            (
                "timestep",
                torch.tensor([timestep_value] * batch_size).to(torch.float32),
            ),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ]
    )
    return sample_unet_inputs


def lcm_inputs(sample_unet_inputs, ref_unet=None):
    """Build the LCM guidance-embedding (``timestep_cond``) trace input.

    The embedding dim comes from ``ref_unet.config.time_cond_proj_dim``. The
    256 fallback preserves the legacy ref-unet-less call shape (the Suite's LCM
    converter calls this with one argument); 256 is correct for every known
    SD1.5 full-distill LCM.
    """
    batch_size = sample_unet_inputs["sample"].shape[0]
    dim = 256 if ref_unet is None else ref_unet.config.time_cond_proj_dim
    return {"timestep_cond": torch.randn(batch_size, dim).to(torch.float32)}


def sdxl_inputs(sample_unet_inputs, ref_unet, model_version):
    sample_shape = sample_unet_inputs["sample"].shape
    batch_size = sample_shape[0]
    h = sample_shape[2] * 8
    w = sample_shape[3] * 8
    original_size = (h, w)
    crops_coords_top_left = (0, 0)

    is_refiner = model_version == ModelVersion.SDXL_REFINER

    if is_refiner:
        aesthetic_score = (6.0,)
        time_ids_list = list(original_size + crops_coords_top_left + aesthetic_score)
    else:
        target_size = (h, w)
        time_ids_list = list(original_size + crops_coords_top_left + target_size)

    time_ids = torch.tensor(time_ids_list).repeat(batch_size, 1).to(torch.int64)
    text_embeds_shape = (
        batch_size,
        get_sdxl_text_embeds_dim(ref_unet, len(time_ids_list)),
    )

    return {
        "time_ids": time_ids,
        "text_embeds": torch.randn(*text_embeds_shape).to(torch.float32),
    }


def get_sdxl_text_embeds_dim(ref_unet, time_ids_dim):
    projection_dim = ref_unet.config.projection_class_embeddings_input_dim
    time_embed_dim = ref_unet.config.addition_time_embed_dim
    return projection_dim - (time_ids_dim * time_embed_dim)


def get_inputs_spec(inputs):
    inputs_spec = {k: (v.shape, v.dtype) for k, v in inputs.items()}
    return inputs_spec


def add_cnet_support(sample_shape, reference_unet):
    additional_residuals_shapes = []

    batch_size = sample_shape[0]
    h, w = sample_shape[2:]

    # conv_in
    out_h, out_w = conv2d_output_shape(
        h,
        w,
        reference_unet.conv_in,
    )
    additional_residuals_shapes.append(
        (batch_size, reference_unet.conv_in.out_channels, out_h, out_w)
    )

    # down_blocks
    for down_block in reference_unet.down_blocks:
        additional_residuals_shapes += [
            (batch_size, resnet.out_channels, out_h, out_w)
            for resnet in down_block.resnets
        ]
        if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
            for downsampler in down_block.downsamplers:
                out_h, out_w = conv2d_output_shape(out_h, out_w, downsampler.conv)
            additional_residuals_shapes.append(
                (
                    batch_size,
                    down_block.downsamplers[-1].conv.out_channels,
                    out_h,
                    out_w,
                )
            )

    # mid_block
    additional_residuals_shapes.append(
        (batch_size, reference_unet.mid_block.resnets[-1].out_channels, out_h, out_w)
    )

    additional_inputs = {}
    for i, shape in enumerate(additional_residuals_shapes):
        sample_residual_input = torch.rand(*shape)
        additional_inputs[f"additional_residual_{i}"] = sample_residual_input

    return additional_inputs


def convert_unet(
    ref_unet,
    model_version: ModelVersion,
    unet_out_path: str,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
    attention_implementation: str = ATTENTION_IMPLEMENTATIONS[0],
    quantize_nbits: str = "none",
):
    coreml_unet = get_unet(model_version, ref_unet, attention_implementation)

    sample_shape = (
        batch_size,  # B
        ref_unet.config.in_channels,  # C
        sample_size[0],  # H
        sample_size[1],  # W
    )

    encoder_hidden_states_shape = get_encoder_hidden_states_shape(ref_unet, batch_size)

    sample_inputs = get_sample_input(
        batch_size, encoder_hidden_states_shape, sample_shape
    )

    if model_version == ModelVersion.LCM:
        if ref_unet.config.time_cond_proj_dim is None:
            raise ValueError(
                "model_version=LCM requires a UNet with a guidance embedding "
                "(config.time_cond_proj_dim), but this checkpoint has none — "
                "it is an LCM-LoRA merge with plain SD1.5 architecture. "
                "Convert it with model_version=SD15 and use an LCM scheduler "
                "at sampling time."
            )
        sample_inputs |= lcm_inputs(sample_inputs, ref_unet)

    if model_version in {ModelVersion.SDXL, ModelVersion.SDXL_REFINER}:
        sample_inputs |= sdxl_inputs(sample_inputs, ref_unet, model_version)

    if controlnet_support:
        sample_inputs |= add_cnet_support(sample_shape, ref_unet)

    sample_inputs_spec = get_inputs_spec(sample_inputs)

    logger.info(f"Sample UNet inputs spec: {sample_inputs_spec}")
    logger.info("JIT tracing..")
    traced_unet = torch.jit.trace(
        coreml_unet, example_inputs=list(sample_inputs.values())
    )
    logger.info("Done.")

    coreml_sample_inputs = get_coreml_inputs(sample_inputs)

    coreml_unet = convert_to_coreml(
        "unet", traced_unet, coreml_sample_inputs, ["noise_pred"], unet_out_path
    )

    del traced_unet
    gc.collect()

    _palettize_and_save(coreml_unet, unet_out_path, quantize_nbits, "unet")


def _palettize_and_save(coreml_model, out_path, quantize_nbits, label):
    """Optionally k-means palettize the weights, then save the ``.mlpackage``.

    The default path (``quantize_nbits="none"``) saves the model untouched. Shared
    by the UNet and the VAE / text-encoder conversions so every component gets the
    same opt-in palettization behaviour and filename-driven cache semantics.
    """
    if quantize_nbits != "none":
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        nbits = int(quantize_nbits)
        logger.info(f"Palettizing {label} weights to {nbits}-bit (kmeans)..")
        t0 = time.time()
        cfg = OptimizationConfig(
            global_config=OpPalettizerConfig(mode="kmeans", nbits=nbits)
        )
        coreml_model = palettize_weights(coreml_model, config=cfg)
        logger.info(f"Palettization took {time.time() - t0:.1f}s")

    coreml_model.save(out_path)
    logger.info(f"Saved {label} into {out_path}")


def convert_vae_decoder(
    ref_vae,
    out_path: str,
    *,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    quantize_nbits: str = "none",
):
    """Convert ``AutoencoderKL``'s decoder (latent -> image) to a ``.mlpackage``.

    ``sample_size`` is the *latent* spatial size (H/8, W/8); the image output is
    8x that. The mid-block self-attention is routed through the ORIGINAL
    (full, fp32-score) processor — diffusers' stock sdpa attention fails to
    convert under coremltools 9, the same reason ``ORIGINAL`` exists for the UNet.
    """
    apply_attention_implementation(ref_vae, "ORIGINAL")
    wrapper = CoreMLVAEDecoderWrapper(ref_vae.eval()).eval()

    latent_shape = (
        batch_size,
        ref_vae.config.latent_channels,
        sample_size[0],
        sample_size[1],
    )
    example = torch.rand(*latent_shape)

    logger.info(f"JIT tracing VAE decoder (latent {latent_shape})..")
    traced = torch.jit.trace(wrapper, example)

    inputs = [ct.TensorType(name="latent", shape=latent_shape, dtype=np.float16)]
    coreml_model = convert_to_coreml("vae_decoder", traced, inputs, ["image"], out_path)
    del traced
    gc.collect()
    _palettize_and_save(coreml_model, out_path, quantize_nbits, "vae_decoder")


def convert_vae_encoder(
    ref_vae,
    out_path: str,
    *,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    quantize_nbits: str = "none",
):
    """Convert ``AutoencoderKL``'s encoder (image -> latent moments) to a ``.mlpackage``.

    ``sample_size`` is the latent spatial size; the image input is 8x that. The
    output ``latent_moments`` is the raw mean‖logvar tensor (2*latent_channels
    channels) — the pipeline samples from it. Attention handling matches the
    decoder.
    """
    apply_attention_implementation(ref_vae, "ORIGINAL")
    wrapper = CoreMLVAEEncoderWrapper(ref_vae.eval()).eval()

    image_shape = (batch_size, 3, sample_size[0] * 8, sample_size[1] * 8)
    example = torch.rand(*image_shape)

    logger.info(f"JIT tracing VAE encoder (image {image_shape})..")
    traced = torch.jit.trace(wrapper, example)

    inputs = [ct.TensorType(name="image", shape=image_shape, dtype=np.float16)]
    coreml_model = convert_to_coreml(
        "vae_encoder", traced, inputs, ["latent_moments"], out_path
    )
    del traced
    gc.collect()
    _palettize_and_save(coreml_model, out_path, quantize_nbits, "vae_encoder")


def convert_text_encoder(
    ckpt_path: str,
    model_version: ModelVersion,
    out_path: str,
    *,
    which: int = 1,
    batch_size: int = 1,
    quantize_nbits: str = "none",
):
    """Convert a CLIP text encoder (token ids -> embeddings) to a ``.mlpackage``.

    ``which`` selects the encoder: 1 = primary (SD1.5/SDXL), 2 = SDXL's second
    (``CLIPTextModelWithProjection``). SDXL uses the penultimate hidden state from
    both encoders plus encoder 2's projected pooled output; SD1.5 uses encoder 1's
    final ``last_hidden_state``. ``input_ids`` crosses the Core ML boundary as
    int32 over the fixed 77-token sequence.
    """
    encoders = load_text_encoders(ckpt_path, model_version)
    if which == 2:
        if len(encoders) < 2:
            raise ValueError(
                f"{model_version.name} has no second text encoder "
                "(text_encoder_2 is SDXL-only)."
            )
        encoder = encoders[1]
        hidden_states_index = -2
        output_pooled = True
        output_names = ["hidden_states", "pooled_embeds"]
    else:
        encoder = encoders[0]
        is_sdxl = model_version in {ModelVersion.SDXL, ModelVersion.SDXL_REFINER}
        hidden_states_index = -2 if is_sdxl else None
        output_pooled = False
        output_names = ["hidden_states"]

    wrapper = CoreMLTextEncoderWrapper(
        encoder.eval(),
        hidden_states_index=hidden_states_index,
        output_pooled=output_pooled,
    ).eval()

    ids_shape = (batch_size, TEXT_TOKEN_SEQUENCE_LENGTH)
    # Trace with int64 (torch embedding gather needs long); declare int32 at the
    # Core ML boundary (coremltools casts the gather index op).
    example = torch.zeros(ids_shape, dtype=torch.int64)

    logger.info(f"JIT tracing text encoder {which} (input_ids {ids_shape})..")
    with static_causal_mask(TEXT_TOKEN_SEQUENCE_LENGTH):
        traced = torch.jit.trace(wrapper, example)

    inputs = [ct.TensorType(name="input_ids", shape=ids_shape, dtype=np.int32)]
    coreml_model = convert_to_coreml(
        f"text_encoder_{which}", traced, inputs, output_names, out_path
    )
    del traced
    gc.collect()
    _palettize_and_save(coreml_model, out_path, quantize_nbits, f"text_encoder_{which}")


def convert(
    ckpt_path: str,
    model_version: ModelVersion,
    out_path: str,
    *,
    component: str = Component.UNET.value,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
    lora_weights: list[tuple[str | os.PathLike, float]] = None,
    attn_impl: str = ATTENTION_IMPLEMENTATIONS[0],
    config_path: str = None,
    quantize_nbits: str = "none",
):
    """Convert a single-file checkpoint component to a Core ML ``.mlpackage``.

    ``component`` selects what to convert (default ``"unet"`` — historical
    behaviour, all UNet-only kwargs apply). ``vae_decoder`` / ``vae_encoder`` /
    ``text_encoder`` / ``text_encoder_2`` convert the corresponding sub-model and
    ignore the UNet-only kwargs (``controlnet_support``, ``lora_weights``,
    ``attn_impl``). Keyword-only past the three required positionals so the package
    can add capabilities without breaking an older caller. Writes ``out_path``;
    returns None.
    """
    if os.path.exists(out_path):
        logger.info(f"Found existing model at {out_path}! Skipping..")
        return

    comp = Component(component)

    if comp is Component.VAE_DECODER:
        convert_vae_decoder(
            load_vae(ckpt_path),
            out_path,
            batch_size=batch_size,
            sample_size=sample_size,
            quantize_nbits=quantize_nbits,
        )
        return

    if comp is Component.VAE_ENCODER:
        convert_vae_encoder(
            load_vae(ckpt_path),
            out_path,
            batch_size=batch_size,
            sample_size=sample_size,
            quantize_nbits=quantize_nbits,
        )
        return

    if comp in {Component.TEXT_ENCODER, Component.TEXT_ENCODER_2}:
        convert_text_encoder(
            ckpt_path,
            model_version,
            out_path,
            which=2 if comp is Component.TEXT_ENCODER_2 else 1,
            batch_size=batch_size,
            quantize_nbits=quantize_nbits,
        )
        return

    if attn_impl not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported attention implementation {attn_impl!r}. "
            f"Expected one of {ATTENTION_IMPLEMENTATIONS}."
        )
    ref_unet = load_unet(ckpt_path, config_path)

    for i, lora_weight in enumerate(lora_weights or []):
        lora_path, strength = lora_weight
        adapter_name = f"lora_{i}"
        ref_unet.load_lora_adapter(lora_path, adapter_name=adapter_name)
        ref_unet.set_adapters([adapter_name], weights=[strength])
        ref_unet.fuse_lora()

    convert_unet(
        ref_unet,
        model_version,
        out_path,
        batch_size,
        sample_size,
        controlnet_support,
        attention_implementation=attn_impl,
        quantize_nbits=quantize_nbits,
    )


def load_unet(ckpt_path, config_path):
    """Load the UNet from a single-file checkpoint, routing on state-dict layout.

    LDM-layout files (the Civitai norm) go through ``from_single_file``.
    Diffusers-layout UNet-only dumps (the canonical full-distill LCM artifacts,
    e.g. ``LCM_Dreamshaper_v7_4k.safetensors``) are rejected by
    ``from_single_file`` outright, so they get a direct state-dict load.
    """
    keys = _safetensors_keys(ckpt_path)
    if keys is not None and is_diffusers_unet_layout(keys):
        return load_unet_from_diffusers_state_dict(ckpt_path)
    return UNet2DConditionModel.from_single_file(
        ckpt_path,
        original_config=config_path,
    )


def _safetensors_keys(ckpt_path):
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


def load_unet_from_diffusers_state_dict(ckpt_path, **config_overrides):
    """Load a diffusers-layout UNet-only safetensors dump (SD1.5-class).

    SD1.5 architecture is assumed (``UNet2DConditionModel`` defaults); the
    cross-attention and guidance-embedding dims are read from the weights, so
    both full-distill LCM dumps (``time_embedding.cond_proj`` present) and
    plain SD1.5 UNet dumps load correctly. A non-SD1.5-class dump fails the
    strict ``load_state_dict`` with an explicit shape/key error.
    ``config_overrides`` exists for tests exercising miniature architectures.
    """
    from safetensors.torch import load_file

    state_dict = load_file(ckpt_path)
    cond_proj = state_dict.get("time_embedding.cond_proj.weight")
    config_kwargs = {
        "sample_size": 64,
        "cross_attention_dim": state_dict[
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
        ].shape[1],
        "time_cond_proj_dim": None if cond_proj is None else cond_proj.shape[1],
    } | config_overrides
    unet = UNet2DConditionModel(**config_kwargs)
    unet.load_state_dict(state_dict)
    return unet


def load_vae(ckpt_path):
    """Load ``AutoencoderKL`` from a single-file checkpoint."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_single_file(ckpt_path)


# Pipeline class whose ``from_single_file`` extracts the text encoder(s) for a
# given model version. SDXL pipelines expose ``text_encoder_2``; SD15/LCM do not.
_TEXT_ENCODER_PIPELINE = {
    ModelVersion.SD15: ("diffusers", "StableDiffusionPipeline"),
    ModelVersion.LCM: ("diffusers", "StableDiffusionPipeline"),
    ModelVersion.SDXL: ("diffusers", "StableDiffusionXLPipeline"),
    ModelVersion.SDXL_REFINER: ("diffusers", "StableDiffusionXLPipeline"),
}


def load_text_encoders(ckpt_path, model_version):
    """Return the checkpoint's CLIP text encoder(s) as a list (1 for SD1.5, 2 for SDXL).

    Loads the matching diffusers pipeline from the single-file checkpoint and
    extracts its text encoder(s). This pulls the whole pipeline (UNet/VAE
    included) — acceptable for an offline, one-shot conversion and the most robust
    way to recover correctly-configured CLIP weights from a single file.
    """
    import importlib

    if model_version not in _TEXT_ENCODER_PIPELINE:
        raise ValueError(f"No text-encoder pipeline mapped for {model_version!r}.")
    module_name, class_name = _TEXT_ENCODER_PIPELINE[model_version]
    pipeline_cls = getattr(importlib.import_module(module_name), class_name)

    pipe = pipeline_cls.from_single_file(ckpt_path)
    encoders = [pipe.text_encoder]
    if getattr(pipe, "text_encoder_2", None) is not None:
        encoders.append(pipe.text_encoder_2)
    return encoders
