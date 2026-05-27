# coreml-diffusion

Convert diffusion-model checkpoints into Core ML `.mlpackage` artifacts for the
Apple Neural Engine (ANE) — framework-free and standalone.

`coreml-diffusion` takes a single-file Stable Diffusion checkpoint and produces a
Core ML UNet you can run on-device (macOS/iOS) via Core ML, in a Python pipeline,
or load into any host that consumes the artifact.

## What this is

A standalone toolkit and knowledge base for running diffusion models on the Apple
Neural Engine via Core ML. The niche is **diffusion on the ANE**: low-power,
GPU-free, embeddable in a Swift/iOS app. ANE is the differentiator — this is about
feasibility and power efficiency for SD1.5/SDXL on ANE, not a raw-throughput claim
against desktop GPUs.

The scope is diffusion architectures generally, not Stable Diffusion specifically.
The project aims to gather, in one place: the conversion path, a reproducible
benchmarking suite for objective comparison, a per-model catalogue documenting the
quirks of each architecture on the ANE, and the sources behind it all.

Supported today: SD1.5 and SDXL (verified). SDXL refiner and LCM convert but are
not yet golden-verified (experimental).

## Install

```sh
uv pip install coreml-diffusion          # from PyPI
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

The output `.mlpackage` is the deliverable: load it natively in Swift/Core ML, run
it through the Python inference pipeline below, or hand it to any consuming host.

### Model sources

Register directories so `--ckpt` accepts a bare name instead of a full path:

```sh
coreml-diffusion sources add comfy /path/to/ComfyUI/models   # --kind comfy|flat
coreml-diffusion sources list                                # sources + checkpoints
coreml-diffusion convert --ckpt v1-5-pruned-emaonly --model-version SD15 --out unet.mlpackage
```

Sources are recorded in `~/.config/coreml-diffusion/sources.toml`. `comfy` knows the
`models/{checkpoints,loras,vae,...}` layout; `flat` is a plain checkpoint directory.

## Library

```python
import coreml_diffusion
from coreml_diffusion import ModelVersion

coreml_diffusion.convert(
    "model.safetensors", ModelVersion.SD15, "unet.mlpackage",
    height=512, width=512, attn_impl="SPLIT_EINSUM",
)
```

## Inference (in progress)

A framework-free inference path lets a converted `.mlpackage` generate images with
no host framework: a `diffusers` pipeline runs the stock VAE / text encoder on
torch while the UNet is served from Core ML on the ANE. This doubles as the
package's own regression anchor — the Tier 2 (`m2`) golden image, asserted on an
Apple Silicon runner — and as the reference for the on-device write-up. See
`tests/m2/`.

## Discovery API

`list_model_versions`, `list_attention_impls`, `list_quant_modes`, and
`CONTRACT_VERSION` report what this build can convert. The identifiers are an
additive-only contract: removing or renaming one is a major version bump, because
downstream consumers reference these strings verbatim.

## ComfyUI

[ComfyUI-CoreMLSuite](https://github.com/aszc-dev/ComfyUI-CoreMLSuite) consumes
this package for its conversion path and drives its node dropdowns from the
discovery API above — installing a newer `coreml-diffusion` surfaces new
conversion types in the node with no Suite change. The Suite is one consumer;
this package neither depends on nor requires ComfyUI.

## License

MIT
