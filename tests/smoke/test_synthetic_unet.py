"""Tier 1 smoke: convert a synthetic micro-UNet through coremltools and confirm
the resulting .mlpackage loads and exposes the I/O contract we declare.

Purpose: catch API breakage in coremltools.convert *without* needing a real SD
checkpoint, the ANE, or a converted model on disk. Runs in seconds on a macOS-ARM
runner.

This is a CONVERSION smoke only — it does not run pipeline inference. End-to-end
inference verification lives in ComfyUI-CoreMLSuite's golden e2e for now; a
package-side custom-inference suite (derived diffusers pipelines) is planned
under tests/inference/.

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""
import platform
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from coreml_diffusion.conversion.unet import CoreMLUNetWrapper


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)


# Tiny shapes — large enough to exercise conv2d + linear + addition kernels in
# coremltools, small enough that conversion finishes in seconds on CPU.
SAMPLE_SHAPE = (1, 4, 8, 8)
TIMESTEP_SHAPE = (1,)
ENCODER_SHAPE = (1, 4, 64)  # native diffusers encoder_hidden_states (batch, tokens, hidden)
OUT_NAME = "noise_pred"


class TinyUNet(nn.Module):
    """Minimal UNet-shaped graph: conv -> add(time+context) -> conv.

    Not a real diffusion model. Just enough op variety to exercise the
    PyTorch -> MIL frontend in coremltools and confirm we can still wire the
    inputs/outputs the way the converter declares them.
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(1, 8)
        self.text_proj = nn.Linear(64, 8)

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
    ):
        h = self.conv_in(sample)
        t_emb = self.time_proj(timestep.unsqueeze(-1)).view(1, 8, 1, 1)
        c_emb = self.text_proj(encoder_hidden_states.mean(1)).view(1, 8, 1, 1)
        h = h + t_emb + c_emb
        return (self.conv_out(h),)


@pytest.fixture(scope="module")
def tiny_mlpackage(tmp_path_factory):
    """Convert TinyUNet once per test session and reuse the .mlpackage."""
    import coremltools as ct

    torch.manual_seed(0)
    model = CoreMLUNetWrapper(
        TinyUNet().eval(),
        SimpleNamespace(name="SD15"),
    )
    example = (
        torch.randn(*SAMPLE_SHAPE),
        torch.randn(*TIMESTEP_SHAPE),
        torch.randn(*ENCODER_SHAPE),
    )
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=SAMPLE_SHAPE, dtype=np.float16),
            ct.TensorType(name="timestep", shape=TIMESTEP_SHAPE, dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=ENCODER_SHAPE, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name=OUT_NAME, dtype=np.float16)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )

    out_dir = tmp_path_factory.mktemp("tiny_unet")
    pkg_path = out_dir / "tiny.mlpackage"
    mlmodel.save(str(pkg_path))
    yield pkg_path
    shutil.rmtree(out_dir, ignore_errors=True)


def test_converted_package_loads_with_declared_io(tiny_mlpackage):
    import coremltools as ct

    # Load the saved .mlpackage back through coremltools (no ComfyUI involved).
    model = ct.models.MLModel(str(tiny_mlpackage))
    spec = model.get_spec()

    input_names = {i.name for i in spec.description.input}
    output_names = {o.name for o in spec.description.output}

    assert input_names == {"sample", "timestep", "encoder_hidden_states"}
    assert OUT_NAME in output_names
