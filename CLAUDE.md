# CLAUDE.md

Guidance for working in this repository.

## What this is

`coreml-diffusion` converts single-file Stable Diffusion checkpoints (SD1.5/SDXL
verified; SDXL refiner + LCM experimental) into Core ML `.mlpackage` UNet
artifacts for the Apple Neural Engine. Extracted from
[ComfyUI-CoreMLSuite](https://github.com/aszc-dev/ComfyUI-CoreMLSuite), which now
depends on this package for its conversion path. Usable as a library, via the
`coreml-diffusion` CLI, or embedded in on-device (iOS) tooling.

The deliverable is the `.mlpackage` on disk. Conversion runs on macOS
(coremltools 9); the package imports and its CLI parse on any platform.

## Hard constraints

- **Framework-free.** The package MUST NOT import `comfy`, `folder_paths`,
  `comfy_extras`, or any ComfyUI module. `import coreml_diffusion` must work in a
  comfy-free environment. Enforced by `tests/unit/test_tier0_purity.py`.
- **Lazy heavy imports.** `coremltools`/`diffusers` are heavy and Mac-oriented.
  Keep them out of the import path of `import coreml_diffusion` and of the CLI's
  `--help`/arg parsing. `convert` is resolved lazily via `__getattr__` in
  `__init__.py`; the CLI defers its heavy import into the command handler. Do not
  add top-level heavy imports to `__init__.py`, `cli.py`, or `tests/unit/`.
- **Discovery contract is additive-only.** `list_model_versions`,
  `list_attention_impls`, `list_quant_modes`, and `CONTRACT_VERSION` form a
  contract consumed by the Suite to populate dropdowns. Saved workflow JSON
  references these identifier strings verbatim. Adding an identifier or promoting
  EXPERIMENTAL→VERIFIED is a minor bump; removing/renaming/demoting is a MAJOR
  bump. See the `__init__.py` docstring.
- **`compose_out_name` output is a cache key.** The `.mlpackage` filename stem is
  the cache key every workflow resolves against. It must stay byte-for-byte
  stable; `tests/unit/test_characterization_out_name.py` locks the behaviour.
- `list_model_versions` returns `.name` (`"SD15"`), not `.value` — the node
  reverses it via `ModelVersion[...]`. Emitting `.value` would KeyError on every
  saved workflow.

## Layout

- `coreml_diffusion/__init__.py` — public surface: discovery API, `Status`,
  `ModelVersion`, lazy `convert`. The `_MODEL_STATUS` map is the single source of
  truth for which conversions the Suite may surface.
- `coreml_diffusion/cli.py` — `coreml-diffusion convert` entry point.
- `coreml_diffusion/convert.py` — conversion mechanics (writes `.mlpackage`,
  stops there; never resolves output paths).
- `coreml_diffusion/conversion/` — `attention.py` (SPLIT_EINSUM etc.),
  `trace.py`, `shapes.py`, `unet.py` (`CoreMLUNetWrapper`).
- `coreml_diffusion/naming.py` — pure `compose_out_name`, `QUANT_NBITS_VALUES`.
- `coreml_diffusion/model_version.py`, `attention.py`, `logger.py` — small leaves.

## Test tiers

Markers auto-applied by directory (`tests/conftest.py`); requesting one tier via
`-m` skips collection of the others, so Tier 0 on Linux never imports the Mac
stack.

- **Tier 0 — `unit`** (Linux, framework-free): `uv run pytest -m unit tests/ -v`
- **Tier 1 — `smoke`** (macOS, synthetic micro-UNet through coremltools):
  `uv run pytest -m smoke tests/ -v`
- **Tier 2 — `m2`** (Apple Silicon + ANE): not yet wired into CI.
- **`inference`** — package-side custom-inference (planned).

CI: `tier0.yml` (Linux unit + ruff), `tier1.yml` (macOS smoke), `publish-pypi.yml`
(Trusted Publishing on GitHub Release; tag must match `pyproject.toml` version).

## Common commands

```sh
uv sync                              # install deps + dev group
uv run pytest -m unit tests/ -v      # Tier 0
uv run ruff check .                  # lint
uv run ruff format --check .         # format check (ruff does not sort imports; lint rule "I" does)
uv build                             # build wheel/sdist
```

## Conventions

- Python 3.12 only (`>=3.12,<3.13`). uv + hatchling.
- No `from __future__ import annotations`.
- Conventional Commits (`feat:`, `fix:`, `chore:`, ...).
- Functional-first, small composable units; don't over-abstract.
- Keep the existing dense module/function docstrings — they carry the contracts.
</content>
</invoke>
