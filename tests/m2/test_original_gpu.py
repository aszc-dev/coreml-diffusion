"""[M2-GPU] ORIGINAL attention correctness gate on the Core ML GPU path.

ORIGINAL is the GPU-targeted attention implementation (SPLIT_EINSUM is the
ANE-friendly default and is what the image golden in test_inference_golden.py
exercises on the ANE). ORIGINAL is retained for broader coremltools coverage and
benchmarks where Core ML on the GPU is a valid target.

This is the real-hardware half of the fp16-overflow fix: the CPU smoke test
(tests/smoke/test_original_attention.py) proves the torch graph upcasts QK^T /
softmax to fp32, but only a converted model run through coremltools' FLOAT16
precision proves that upcast *survives* to the GPU. If it were downcast away,
SD1.5 self-attention at 64x64 (4096 query tokens) would overflow fp16 -> NaN.

Convert a real SD1.5 UNet with ORIGINAL, run a single forward on the GPU, and
gate on: output is finite (the overflow guard) AND cosine similarity to a torch
fp32 reference UNet stays high (the model still computes the right thing).

Requires Apple Silicon and a single-file SD1.5 checkpoint:
  COREML_DIFFUSION_TEST_CKPT   absolute path to the checkpoint
  ORIGINAL_GPU_COSINE_MIN      optional cosine floor (default 0.99)
Skips otherwise, so Tier 0 on Linux is unaffected.
"""

import os
import platform

import numpy as np
import pytest

CKPT = os.environ.get("COREML_DIFFUSION_TEST_CKPT")
COSINE_MIN = float(os.environ.get("ORIGINAL_GPU_COSINE_MIN", "0.99"))

SAMPLE_SHAPE = (1, 4, 64, 64)  # 512x512 latent — the fp16-overflow case
TIMESTEP = 999.0
TOKENS = 77
SEED = 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def prerequisites():
    if platform.machine() != "arm64":
        pytest.skip("requires Apple Silicon")
    if not CKPT:
        pytest.skip(
            "set COREML_DIFFUSION_TEST_CKPT to a single-file SD1.5 checkpoint "
            "to run the Tier 2 ORIGINAL/GPU correctness gate"
        )


def test_original_attention_survives_coreml_to_gpu(prerequisites, tmp_path):
    import coremltools as ct
    import torch

    from coreml_diffusion import ModelVersion
    from coreml_diffusion.convert import convert, load_unet

    out_path = tmp_path / "sd15_original_b1.mlpackage"
    convert(
        CKPT,
        ModelVersion.SD15,
        str(out_path),
        batch_size=1,
        sample_size=(64, 64),
        attn_impl="ORIGINAL",
    )

    # Fixed input shared by both the converted model and the torch reference.
    torch.manual_seed(SEED)
    sample = torch.randn(*SAMPLE_SHAPE)
    timestep = torch.tensor([TIMESTEP])
    cross_dim = load_unet(CKPT, None).config.cross_attention_dim
    encoder_hidden_states = torch.randn(1, TOKENS, cross_dim)

    # Ground truth: the canonical UNet math in fp32 on torch (stock diffusers).
    ref_unet = load_unet(CKPT, None).eval()
    with torch.no_grad():
        reference = ref_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0].numpy()

    # Converted ORIGINAL UNet on the Core ML GPU path (fp16 weights/activations,
    # attention upcast to fp32 inside the graph).
    model = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    predicted = model.predict(
        {
            "sample": sample.numpy().astype(np.float16),
            "timestep": timestep.numpy().astype(np.float16),
            "encoder_hidden_states": encoder_hidden_states.numpy().astype(np.float16),
        }
    )["noise_pred"]

    assert np.isfinite(predicted).all(), "fp16 attention overflowed to NaN/inf"
    cosine = _cosine(predicted, reference)
    assert cosine >= COSINE_MIN, (
        f"cosine {cosine:.5f} < {COSINE_MIN}: ORIGINAL/GPU output diverged from "
        f"the torch fp32 reference"
    )
