# coreml-diffusion

Convert diffusion-model checkpoints into Core ML `.mlpackage` artifacts for the
Apple Neural Engine (ANE) — framework-free and independent of ComfyUI.

`coreml-diffusion` takes a single-file Stable Diffusion checkpoint and produces a
Core ML UNet you can run on-device (macOS/iOS) or load back into ComfyUI via
[ComfyUI-CoreMLSuite](https://github.com/aszc-dev/ComfyUI-CoreMLSuite), which
depends on this package for its conversion path.

## Positioning

The niche is **diffusion models on the Apple Neural Engine via Core ML** — inside
ComfyUI and on-device. ANE is the differentiator: low-power, GPU-free, embeddable
in a Swift/iOS app. This is about feasibility and power efficiency for SD1.5/SDXL
on ANE, not a raw-throughput claim against desktop GPUs.

Supported today: SD1.5 and SDXL (verified). SDXL refiner and LCM convert but are
not yet golden-verified (experimental). The scope is diffusion architectures
generally, not Stable Diffusion specifically.

## Install

```sh
uv pip install coreml-diffusion          # from PyPI (planned)
uv pip install -e .                       # from a checkout
```

Requires Python 3.12 and (for conversion) `coremltools` 9 — conversion runs on
macOS; the package imports and its CLI parse on any platform.

## CLI

```sh
coreml-diffusion convert \
    --ckpt path/to/model.safetensors \
    --model-version SD15 \
    --out unet.mlpackage \
    --height 512 --width 512 \
    --attn-impl SPLIT_EINSUM \
    --quantize none
```

Options: `--batch-size`, `--controlnet`, `--lora PATH[:STRENGTH]` (repeatable),
`--config` (original-config YAML). `--quantize {none,8,6,4}` applies k-means
weight palettization. Run `coreml-diffusion convert --help` for the full list.

The output `.mlpackage` is the deliverable: load it natively in Swift/Core ML, or
through ComfyUI-CoreMLSuite.

## Library

```python
import coreml_diffusion
from coreml_diffusion import ModelVersion

coreml_diffusion.convert(
    "model.safetensors", ModelVersion.SD15, "unet.mlpackage",
    height=512, width=512, attn_impl="SPLIT_EINSUM",
)
```

Discovery API (`list_model_versions`, `list_attention_impls`, `list_quant_modes`,
`CONTRACT_VERSION`) reports what this build can convert; the identifiers are an
additive-only contract (removing/renaming one is a major bump).

## License

MIT
