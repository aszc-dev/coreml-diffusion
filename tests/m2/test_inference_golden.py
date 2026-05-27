"""[M2-ANE] golden-image anchor for the comfy-free inference pipeline.

Generates an image end-to-end with NO ComfyUI: a stock ``diffusers`` pipeline
runs the VAE / text encoder on torch while the converted UNet runs on the Apple
Neural Engine (``coreml_diffusion.build_pipeline``). The result is asserted
against a stored golden via a two-tier gate, mirroring the Suite's Tier 2:

  - exact pixel match (strict fast-path), OR
  - PSNR >= GOLDEN_PSNR_MIN_DB (soft gate).

The ANE is nondeterministic: the same model and seed drift several dB run to run
as the sampling steps amplify per-step UNet differences (kernel selection / fp
accumulation order). The Suite observed ~23 dB same-scene, so the 20 dB default
leaves margin while still catching a broken image (which lands far lower).
Override with GOLDEN_PSNR_MIN_DB — raise it for pure-refactor PRs that must not
change the math, lower it for toolchain bumps.

Requires Apple Silicon + the ANE and a UNet converted at batch=2 (CFG feeds
uncond+cond in one forward pass). Skips otherwise, so Tier 0 on Linux is
unaffected. Prerequisites:
  COREML_DIFFUSION_TEST_CKPT       single-file SD1.5 checkpoint
  COREML_DIFFUSION_TEST_MLPACKAGE  its UNet converted at batch=2, 512x512
The first run with no golden writes one and fails so it can be reviewed before
being committed.
"""

import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import pytest

CKPT = os.environ.get("COREML_DIFFUSION_TEST_CKPT")
MLPACKAGE = os.environ.get("COREML_DIFFUSION_TEST_MLPACKAGE")

PROMPT = "a photograph of an astronaut riding a horse"
SEED = 0
STEPS = 20
GUIDANCE = 7.5
PSNR_MIN_DB = float(os.environ.get("GOLDEN_PSNR_MIN_DB", "20"))
GOLDEN_DIR = Path(__file__).parent / "goldens"
GOLDEN_PNG = GOLDEN_DIR / "sd15_astronaut.png"
GOLDEN_SHA = GOLDEN_DIR / "sd15_astronaut.sha256"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


@pytest.fixture
def prerequisites():
    if platform.machine() != "arm64":
        pytest.skip("requires Apple Silicon + ANE")
    if not (CKPT and MLPACKAGE):
        pytest.skip(
            "set COREML_DIFFUSION_TEST_CKPT and COREML_DIFFUSION_TEST_MLPACKAGE "
            "(a batch=2 512x512 SD1.5 UNet) to run the Tier 2 golden"
        )


def _generate_image():
    import torch

    import coreml_diffusion
    from coreml_diffusion import ModelVersion

    pipe = coreml_diffusion.build_pipeline(CKPT, MLPACKAGE, ModelVersion.SD15)
    return pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=512,
        width=512,
        generator=torch.manual_seed(SEED),
    ).images[0]


def test_sd15_astronaut_matches_golden(prerequisites):
    from PIL import Image

    image = _generate_image()

    if not (GOLDEN_PNG.exists() and GOLDEN_SHA.exists()):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        image.save(GOLDEN_PNG)
        sha = hashlib.sha256(GOLDEN_PNG.read_bytes()).hexdigest()
        GOLDEN_SHA.write_text(sha + "\n")
        pytest.fail(
            f"No golden present; wrote {GOLDEN_PNG.name} + {GOLDEN_SHA.name}. "
            f"Review the image and commit it, then re-run."
        )

    produced = np.asarray(image.convert("RGB"))
    golden = np.asarray(Image.open(GOLDEN_PNG).convert("RGB"))
    if produced.shape != golden.shape:
        pytest.fail(f"shape mismatch: golden={golden.shape} actual={produced.shape}")

    # Strict gate: identical pixels (a refactor that doesn't touch the math
    # should hit this). Soft gate: PSNR absorbs ANE run-to-run drift.
    if np.array_equal(produced, golden):
        return
    psnr = _psnr(produced, golden)
    assert psnr >= PSNR_MIN_DB, (
        f"PSNR {psnr:.2f} dB < {PSNR_MIN_DB} dB threshold "
        f"(ANE drift or a real regression)"
    )
