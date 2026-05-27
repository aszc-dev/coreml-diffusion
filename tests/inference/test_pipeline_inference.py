"""Package-side custom-inference suite (Tier: inference).

Runs a converted ``.mlpackage`` UNet through a stock ``diffusers`` pipeline with
no ComfyUI (``coreml_diffusion.build_pipeline``), generating an image end-to-end
on the ANE. This produces the package's OWN golden image — distinct from the
Suite's comfy e2e (different sampler), so divergence from that is expected, not a
regression.

Prerequisites (the test skips unless all are set):
  COREML_DIFFUSION_TEST_CKPT        single-file SD1.5 .safetensors
  COREML_DIFFUSION_TEST_MLPACKAGE   its UNet converted at batch_size=2, 512x512
                                    (batch=2 because guided CFG feeds uncond+cond)
On first run with no golden present, the image is written as the golden and the
PSNR assertion is skipped; commit it, then later runs assert against it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

CKPT = os.environ.get("COREML_DIFFUSION_TEST_CKPT")
MLPACKAGE = os.environ.get("COREML_DIFFUSION_TEST_MLPACKAGE")

pytestmark = pytest.mark.skipif(
    not (CKPT and MLPACKAGE),
    reason="set COREML_DIFFUSION_TEST_CKPT and COREML_DIFFUSION_TEST_MLPACKAGE "
    "(a batch=2 512x512 SD1.5 UNet) to run the inference golden",
)

PROMPT = "a photograph of an astronaut riding a horse"
SEED = 0
STEPS = 20
GUIDANCE = 7.5
PSNR_THRESHOLD = 25.0
GOLDEN = Path(__file__).parent / "golden" / "sd15_astronaut.npy"


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / mse)


def test_converted_unet_generates_expected_latent():
    import torch

    import coreml_diffusion
    from coreml_diffusion import ModelVersion

    pipe = coreml_diffusion.build_pipeline(CKPT, MLPACKAGE, ModelVersion.SD15)
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=512,
        width=512,
        generator=torch.manual_seed(SEED),
    ).images[0]
    produced = np.asarray(image)

    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN, produced)
        pytest.skip(
            f"captured golden at {GOLDEN}; commit it, then this run asserts PSNR"
        )

    golden = np.load(GOLDEN)
    assert produced.shape == golden.shape, (produced.shape, golden.shape)
    psnr = _psnr(produced, golden)
    assert psnr > PSNR_THRESHOLD, (
        f"PSNR {psnr:.1f} dB below {PSNR_THRESHOLD} dB threshold"
    )
