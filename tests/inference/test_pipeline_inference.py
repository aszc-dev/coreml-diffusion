"""Placeholder for the package-side custom-inference suite (Tier: inference).

Planned: run inference through derived `diffusers` pipelines with custom behavior
so a converted `.mlpackage` can generate an image end-to-end WITHOUT ComfyUI, and
showcase results. That path differs from ComfyUI's sampler, so it will produce its
own golden image (an expected divergence from the Suite's comfy e2e, to be
re-anchored here — not a regression).

Until this suite exists, inference verification relies on ComfyUI-CoreMLSuite's
golden e2e (`pytest -m m2`). This module is intentionally skipped; it reserves the
directory, the `inference` marker, and the intent.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="custom-inference suite not implemented yet; inference verified via "
    "ComfyUI-CoreMLSuite golden e2e for now"
)


def test_converted_unet_generates_expected_latent():
    # TODO: build a derived diffusers pipeline around a converted .mlpackage UNet,
    # run a fixed-seed denoise, and assert the decoded image matches a package-side
    # golden (PSNR threshold). Capture that golden when the pipeline lands.
    raise NotImplementedError
