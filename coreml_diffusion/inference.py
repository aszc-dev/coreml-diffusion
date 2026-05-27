"""Framework-free inference for a converted Core ML UNet.

Runs a `.mlpackage` UNet inside a stock ``diffusers`` pipeline with NO ComfyUI:
the VAE and text encoder run on torch (CPU/MPS) while the UNet is served from
Core ML on the ANE. This is the package-side counterpart to ``convert`` — it
proves a converted artifact generates an image end-to-end, and provides the
golden anchor for the ``inference`` test tier.

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
    compute_unit=DEFAULT_COMPUTE_UNIT,
    torch_device="cpu",
    **from_single_file_kwargs,
):
    """Load a stock diffusers pipeline from ``ckpt_path`` and swap in the Core ML UNet.

    The VAE / text encoder / scheduler come from the same checkpoint and run on
    ``torch_device``; only the UNet is served from ``mlpackage_path`` via Core ML.
    Returns the pipeline ready to call. Wired for SD15 and SDXL; the golden
    anchor that verifies output is captured by the ``inference`` test tier.
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
    return pipe
