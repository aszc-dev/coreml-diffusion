"""[M2-ANE] golden-image anchors for the comfy-free inference pipeline.

Generates an image end-to-end with NO ComfyUI, in two variants, each asserted
against its own stored golden via a two-tier gate (mirroring the Suite's Tier 2):

  - exact pixel match (strict fast-path), OR
  - PSNR >= GOLDEN_PSNR_MIN_DB (soft gate).

1. **UNet-only** (``test_sd15_astronaut_matches_golden``): the stock ``diffusers``
   pipeline runs the VAE / text encoder on torch; only the converted UNet runs on
   the ANE. Needs CKPT + MLPACKAGE.
2. **Full Core ML** (``test_sd15_astronaut_full_coreml_matches_golden``): the VAE
   decoder and CLIP text encoder are ALSO converted and swapped, so the whole
   forward runs on Core ML — no torch UNet/VAE/CLIP, no Apple ml-stable-diffusion.
   This is the regression anchor for the all-components-replaced path (which the
   UNet-only variant cannot exercise: with a real VAE still present, the pipeline
   never hits e.g. the ``.device`` lookup that only Core ML stand-ins must answer).
   Needs CKPT + MLPACKAGE + VAE_DECODER + TEXT_ENCODER.

The ANE is nondeterministic: the same model and seed drift several dB run to run
as the sampling steps amplify per-step UNet differences (kernel selection / fp
accumulation order). The Suite observed ~23 dB same-scene, so the 20 dB default
leaves margin while still catching a broken image (which lands far lower).
Override with GOLDEN_PSNR_MIN_DB — raise it for pure-refactor PRs that must not
change the math, lower it for toolchain bumps. The two variants have *different*
goldens: the full-Core ML path serves the VAE/CLIP through fp16 Core ML instead
of torch, so the image is not pixel-identical to the UNet-only one.

Requires Apple Silicon + the ANE and a UNet converted at batch=2 (CFG feeds
uncond+cond in one forward pass). Skips otherwise, so Tier 0 on Linux is
unaffected. Prerequisites (env vars, absolute paths):
  COREML_DIFFUSION_TEST_CKPT          single-file SD1.5 checkpoint
  COREML_DIFFUSION_TEST_MLPACKAGE     its UNet converted at batch=2, 512x512
  COREML_DIFFUSION_TEST_VAE_DECODER   its VAE decoder (component=vae_decoder, 512)
  COREML_DIFFUSION_TEST_TEXT_ENCODER  its CLIP text encoder (component=text_encoder)
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
VAE_DECODER = os.environ.get("COREML_DIFFUSION_TEST_VAE_DECODER")
TEXT_ENCODER = os.environ.get("COREML_DIFFUSION_TEST_TEXT_ENCODER")

PROMPT = "a photograph of an astronaut riding a horse"
SEED = 0
STEPS = 20
GUIDANCE = 7.5
PSNR_MIN_DB = float(os.environ.get("GOLDEN_PSNR_MIN_DB", "20"))
GOLDEN_DIR = Path(__file__).parent / "goldens"
GOLDEN_PNG = GOLDEN_DIR / "sd15_astronaut.png"
GOLDEN_SHA = GOLDEN_DIR / "sd15_astronaut.sha256"
FULL_GOLDEN_PNG = GOLDEN_DIR / "sd15_astronaut_full_coreml.png"
FULL_GOLDEN_SHA = GOLDEN_DIR / "sd15_astronaut_full_coreml.sha256"


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


@pytest.fixture
def full_coreml_prerequisites(prerequisites):
    if not (VAE_DECODER and TEXT_ENCODER):
        pytest.skip(
            "set COREML_DIFFUSION_TEST_VAE_DECODER and "
            "COREML_DIFFUSION_TEST_TEXT_ENCODER (a 512 vae_decoder + a text_encoder "
            "from the same checkpoint) to run the full-Core ML golden"
        )


def _generate_image(**build_kwargs):
    import torch

    import coreml_diffusion
    from coreml_diffusion import ModelVersion

    pipe = coreml_diffusion.build_pipeline(
        CKPT, MLPACKAGE, ModelVersion.SD15, **build_kwargs
    )
    return pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=512,
        width=512,
        generator=torch.manual_seed(SEED),
    ).images[0]


def _assert_matches_golden(image, golden_png: Path, golden_sha: Path):
    from PIL import Image

    if not (golden_png.exists() and golden_sha.exists()):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        image.save(golden_png)
        golden_sha.write_text(
            hashlib.sha256(golden_png.read_bytes()).hexdigest() + "\n"
        )
        pytest.fail(
            f"No golden present; wrote {golden_png.name} + {golden_sha.name}. "
            f"Review the image and commit it, then re-run."
        )

    produced = np.asarray(image.convert("RGB"))
    golden = np.asarray(Image.open(golden_png).convert("RGB"))
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


def test_sd15_astronaut_matches_golden(prerequisites):
    image = _generate_image()
    _assert_matches_golden(image, GOLDEN_PNG, GOLDEN_SHA)


def test_sd15_astronaut_full_coreml_matches_golden(full_coreml_prerequisites):
    image = _generate_image(
        vae_decoder_mlpackage=VAE_DECODER,
        text_encoder_mlpackage=TEXT_ENCODER,
    )
    _assert_matches_golden(image, FULL_GOLDEN_PNG, FULL_GOLDEN_SHA)
