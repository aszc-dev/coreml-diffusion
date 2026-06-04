"""Framework-free inference for a converted Core ML UNet.

Runs a `.mlpackage` UNet inside a stock ``diffusers`` pipeline with NO ComfyUI:
the VAE and text encoder run on torch (CPU/MPS) while the UNet is served from
Core ML on the ANE. This is the package-side counterpart to ``convert`` — it
proves a converted artifact generates an image end-to-end, and provides the
Tier 2 (``m2``) golden anchor (``tests/m2/``).

``CoreMLUNet`` is the inverse of ``conversion.unet.CoreMLUNetWrapper``: the
wrapper adapts diffusers UNet *inputs* into the flat Core ML tensor contract at
convert time; this adapter unpacks a diffusers call back into that same flat
contract at run time, feeds it to ``coremltools``, and re-wraps the output as a
diffusers ``UNet2DConditionOutput``.

Core ML input contract (see ``convert.convert_unet``): every input is float16.
  sample (B,C,H,W), timestep (B,), encoder_hidden_states (B,77,cross_dim);
  + LCM:  timestep_cond (B,256)
  + SDXL: time_ids (B,6), text_embeds (B,proj_dim)
  + ctrl: additional_residual_0..N
Single output ``noise_pred`` (float32).

The traced model has FIXED input shapes: the batch and resolution are baked in
at conversion. Classifier-free guidance feeds a batch of 2 (uncond+cond), so the
artifact must be converted with ``batch_size=2`` to run a guided pipeline in one
forward pass. Height/width must match the converted ``sample_size``.
"""

import numpy as np
import torch

from coreml_diffusion.logger import logger
from coreml_diffusion.model_version import ModelVersion

# Compute-unit default: let Core ML place ops on the ANE (falling back to CPU for
# unsupported ops). Override per call for A/B against CPU_ONLY / CPU_AND_GPU.
DEFAULT_COMPUTE_UNIT = "CPU_AND_NE"


def _f16(tensor):
    """diffusers tensor -> contiguous float16 numpy, matching the trace dtype."""
    return np.ascontiguousarray(tensor.detach().to(torch.float16).cpu().numpy())


def _i32(tensor):
    """token-id tensor -> contiguous int32 numpy, matching the trace dtype."""
    return np.ascontiguousarray(tensor.detach().to(torch.int32).cpu().numpy())


def _resolve_unit(compute_unit):
    import coremltools as ct

    if isinstance(compute_unit, ct.ComputeUnit):
        return compute_unit
    return ct.ComputeUnit[compute_unit]


class CoreMLUNet(torch.nn.Module):
    """A ``UNet2DConditionModel`` stand-in backed by a Core ML ``.mlpackage``.

    Carries the reference UNet's ``config`` so the surrounding diffusers pipeline
    reads the right ``in_channels`` / ``addition_embed_type`` / ``time_cond_proj_dim``
    etc., but runs the forward pass through coremltools instead of torch.
    """

    def __init__(
        self, mlpackage_path, ref_unet, model_version, compute_unit=DEFAULT_COMPUTE_UNIT
    ):
        super().__init__()
        import coremltools as ct

        self.config = ref_unet.config
        self.dtype = torch.float16
        self.model_version = model_version

        unit = (
            compute_unit
            if isinstance(compute_unit, ct.ComputeUnit)
            else ct.ComputeUnit[compute_unit]
        )
        logger.info(f"Loading {mlpackage_path} to {unit.name}")
        self.model = ct.models.MLModel(mlpackage_path, compute_units=unit)
        self._output_name = self.model.get_spec().description.output[0].name

    @property
    def device(self):
        # Inputs/outputs are materialised on CPU; the math runs on the ANE via
        # coremltools. A plain nn.Module has no `.device`, but diffusers'
        # `pipeline.device` / `_execution_device` walk the component modules for
        # one — required once every component is a Core ML stand-in.
        return torch.device("cpu")

    @property
    def _is_lcm(self):
        return self.model_version is ModelVersion.LCM

    @property
    def _is_sdxl(self):
        return self.model_version in {ModelVersion.SDXL, ModelVersion.SDXL_REFINER}

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        timestep_cond=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict=True,
        **_ignored,
    ):
        batch = sample.shape[0]

        # timestep arrives as a python scalar, 0-dim, or (1,) tensor; the trace
        # baked a (B,) timestep, so broadcast to the batch.
        ts = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)
        if ts.numel() == 1:
            ts = ts.expand(batch)

        inputs = {
            "sample": _f16(sample),
            "timestep": _f16(ts),
            "encoder_hidden_states": _f16(encoder_hidden_states),
        }

        if self._is_lcm:
            if timestep_cond is None:
                raise ValueError(
                    "LCM UNet requires timestep_cond (guidance embedding)."
                )
            inputs["timestep_cond"] = _f16(timestep_cond)

        if self._is_sdxl:
            if not added_cond_kwargs or "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    "SDXL UNet requires added_cond_kwargs with time_ids/text_embeds."
                )
            inputs["time_ids"] = _f16(added_cond_kwargs["time_ids"])
            inputs["text_embeds"] = _f16(added_cond_kwargs["text_embeds"])

        if down_block_additional_residuals is not None:
            residuals = list(down_block_additional_residuals)
            if mid_block_additional_residual is not None:
                residuals.append(mid_block_additional_residual)
            for i, residual in enumerate(residuals):
                inputs[f"additional_residual_{i}"] = _f16(residual)

        prediction = self.model.predict(inputs)[self._output_name]
        noise_pred = torch.from_numpy(np.ascontiguousarray(prediction)).to(
            sample.device
        )

        if not return_dict:
            return (noise_pred,)
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

        return UNet2DConditionOutput(sample=noise_pred)


class CoreMLVAE(torch.nn.Module):
    """An ``AutoencoderKL`` stand-in backed by Core ML ``.mlpackage`` artifacts.

    Serves ``decode`` (latent -> image) and, when an encoder package is supplied,
    ``encode`` (image -> latent distribution). Carries the reference VAE's
    ``config`` so the pipeline reads ``scaling_factor`` / ``block_out_channels``
    etc.; the actual math runs through coremltools.

    ``dtype`` is reported as float32 on purpose: SDXL's pipeline upcasts the VAE to
    fp32 when ``dtype == float16 and config.force_upcast`` — that path pokes at
    ``vae.decoder`` / ``post_quant_conv`` submodules this adapter does not have.
    Reporting float32 skips it; the converted package runs fp16 internally either
    way (the scale was baked at conversion).
    """

    def __init__(
        self,
        ref_vae,
        *,
        decoder_mlpackage=None,
        encoder_mlpackage=None,
        compute_unit=DEFAULT_COMPUTE_UNIT,
    ):
        super().__init__()
        import coremltools as ct

        if decoder_mlpackage is None and encoder_mlpackage is None:
            raise ValueError("CoreMLVAE needs a decoder and/or an encoder package.")

        self.config = ref_vae.config
        self.dtype = torch.float32
        unit = _resolve_unit(compute_unit)

        self._decoder = None
        self._decoder_out = None
        if decoder_mlpackage is not None:
            logger.info(f"Loading VAE decoder {decoder_mlpackage} to {unit.name}")
            self._decoder = ct.models.MLModel(decoder_mlpackage, compute_units=unit)
            self._decoder_out = self._decoder.get_spec().description.output[0].name

        self._encoder = None
        self._encoder_out = None
        if encoder_mlpackage is not None:
            logger.info(f"Loading VAE encoder {encoder_mlpackage} to {unit.name}")
            self._encoder = ct.models.MLModel(encoder_mlpackage, compute_units=unit)
            self._encoder_out = self._encoder.get_spec().description.output[0].name

    @property
    def device(self):
        return torch.device("cpu")  # see CoreMLUNet.device

    def decode(self, z, return_dict=True, **_ignored):
        if self._decoder is None:
            raise RuntimeError("CoreMLVAE was built without a decoder package.")
        prediction = self._decoder.predict({"latent": _f16(z)})[self._decoder_out]
        sample = torch.from_numpy(np.ascontiguousarray(prediction)).to(z.device)
        if not return_dict:
            return (sample,)
        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=sample)

    def encode(self, x, return_dict=True, **_ignored):
        if self._encoder is None:
            raise RuntimeError("CoreMLVAE was built without an encoder package.")
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        prediction = self._encoder.predict({"image": _f16(x)})[self._encoder_out]
        moments = torch.from_numpy(np.ascontiguousarray(prediction)).to(x.device)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        return AutoencoderKLOutput(latent_dist=posterior)


class _CoreMLTextEncoderOutput:
    """The subset of a CLIP encoder's output the diffusers pipelines actually read.

    ``out[0]`` is the pooled vector when the package emits one (SDXL encoder 2),
    otherwise the embeddings (SD1.5 reads the final hidden state here). ``out.
    hidden_states[-2]`` is the penultimate state SDXL concatenates — the converter
    bakes exactly that tensor, so a 2-tuple whose ``[-2]`` is it satisfies the
    access without shipping every layer's state.
    """

    def __init__(self, embeds, pooled):
        self.last_hidden_state = embeds
        self.text_embeds = pooled
        self.pooler_output = pooled
        self.hidden_states = (embeds, embeds)
        self._first = embeds if pooled is None else pooled

    def __getitem__(self, idx):
        if idx == 0:
            return self._first
        raise IndexError(f"CoreML text-encoder output exposes only [0]; got {idx}")


class CoreMLTextEncoder(torch.nn.Module):
    """A CLIP text-encoder stand-in backed by a Core ML ``.mlpackage``.

    Token ids in (int32 at the boundary), embeddings out, wrapped so the diffusers
    pipelines' ``encode_prompt`` reads it transparently. Carries the reference
    encoder's ``config`` (the pipeline inspects ``use_attention_mask`` /
    ``projection_dim``). ``clip_skip`` is fixed at conversion (SD1.5 final state,
    SDXL penultimate); requesting a different skip at run time is unsupported.
    """

    def __init__(
        self, mlpackage_path, ref_text_encoder, *, compute_unit=DEFAULT_COMPUTE_UNIT
    ):
        super().__init__()
        import coremltools as ct

        self.config = ref_text_encoder.config
        self.dtype = torch.float16
        unit = _resolve_unit(compute_unit)
        logger.info(f"Loading text encoder {mlpackage_path} to {unit.name}")
        self.model = ct.models.MLModel(mlpackage_path, compute_units=unit)
        names = {o.name for o in self.model.get_spec().description.output}
        self._pooled_name = "pooled_embeds" if "pooled_embeds" in names else None

    @property
    def device(self):
        return torch.device("cpu")  # see CoreMLUNet.device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=None,
        return_dict=None,
        **_ignored,
    ):
        prediction = self.model.predict({"input_ids": _i32(input_ids)})
        embeds = torch.from_numpy(np.ascontiguousarray(prediction["hidden_states"])).to(
            input_ids.device
        )
        pooled = None
        if self._pooled_name is not None:
            pooled = torch.from_numpy(
                np.ascontiguousarray(prediction[self._pooled_name])
            ).to(input_ids.device)
        return _CoreMLTextEncoderOutput(embeds, pooled)


# Stock diffusers pipeline per verified model. Experimental versions are convertible
# but their pipelines are not wired here yet.
_PIPELINE_IMPORTS = {
    ModelVersion.SD15: ("diffusers", "StableDiffusionPipeline"),
    ModelVersion.SDXL: ("diffusers", "StableDiffusionXLPipeline"),
}


def build_pipeline(
    ckpt_path,
    mlpackage_path,
    model_version,
    *,
    vae_decoder_mlpackage=None,
    vae_encoder_mlpackage=None,
    text_encoder_mlpackage=None,
    text_encoder_2_mlpackage=None,
    compute_unit=DEFAULT_COMPUTE_UNIT,
    vae_compute_unit=None,
    text_encoder_compute_unit=None,
    torch_device="cpu",
    **from_single_file_kwargs,
):
    """Load a stock diffusers pipeline from ``ckpt_path`` and swap in Core ML parts.

    The UNet (``mlpackage_path``) is always served from Core ML. Any component for
    which a ``.mlpackage`` is supplied is swapped too, so the whole forward can run
    on Core ML / ANE; the rest stay on torch (``torch_device``). Passing only
    ``mlpackage_path`` reproduces the original UNet-only behaviour.

    ``vae_compute_unit`` / ``text_encoder_compute_unit`` override the placement of
    those components (VAE is often faster on the GPU, e.g. ``"CPU_AND_GPU"``);
    each defaults to ``compute_unit``. Wired for SD15 and SDXL; the golden anchor
    verifying output is the Tier 2 (``m2``) test tier.
    """
    if model_version not in _PIPELINE_IMPORTS:
        raise NotImplementedError(
            f"No inference pipeline wired for {model_version.name}; "
            f"supported: {[v.name for v in _PIPELINE_IMPORTS]}."
        )
    import importlib

    module_name, class_name = _PIPELINE_IMPORTS[model_version]
    pipeline_cls = getattr(importlib.import_module(module_name), class_name)

    pipe = pipeline_cls.from_single_file(ckpt_path, **from_single_file_kwargs)
    pipe.to(torch_device)

    pipe.unet = CoreMLUNet(mlpackage_path, pipe.unet, model_version, compute_unit)

    if vae_decoder_mlpackage is not None or vae_encoder_mlpackage is not None:
        pipe.vae = CoreMLVAE(
            pipe.vae,
            decoder_mlpackage=vae_decoder_mlpackage,
            encoder_mlpackage=vae_encoder_mlpackage,
            compute_unit=vae_compute_unit or compute_unit,
        )

    te_unit = text_encoder_compute_unit or compute_unit
    if text_encoder_mlpackage is not None:
        pipe.text_encoder = CoreMLTextEncoder(
            text_encoder_mlpackage, pipe.text_encoder, compute_unit=te_unit
        )
    if text_encoder_2_mlpackage is not None:
        if getattr(pipe, "text_encoder_2", None) is None:
            raise ValueError(
                f"{model_version.name} pipeline has no text_encoder_2 to replace."
            )
        pipe.text_encoder_2 = CoreMLTextEncoder(
            text_encoder_2_mlpackage, pipe.text_encoder_2, compute_unit=te_unit
        )

    return pipe
