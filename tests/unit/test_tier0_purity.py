"""Gate: prove the Tier-0 lane is framework-free.

In a pure ``pytest -m unit`` run, none of the heavy/Mac-only modules
(coremltools, diffusers) nor any ComfyUI module may be in sys.modules after
collection. If they are, a tests/unit/ file is transitively importing them and
the Tier-0 promise — "runs on Linux with no Mac stack" — is broken.

When other tiers are also collected (smoke pulls coremltools deliberately), the
check is skipped, so it is only meaningful in a pure ``-m unit`` run.
"""
import sys

import pytest

BANNED_ROOTS = {
    "comfy",
    "comfy_extras",
    "folder_paths",
    "nodes",
    "coremltools",
    "diffusers",
}


def test_no_heavy_modules_loaded_by_unit_tier(request):
    markexpr = request.config.option.markexpr
    if markexpr != "unit":
        pytest.skip(
            "purity gate only meaningful in a pure `-m unit` run "
            f"(got markexpr={markexpr!r}); other tiers import coremltools/diffusers."
        )
    loaded = {name for name in sys.modules if name.split(".")[0] in BANNED_ROOTS}
    assert not loaded, (
        f"Tier-0 leakage: these heavy/framework modules are in sys.modules after "
        f"collecting tests/unit/: {sorted(loaded)}. Framework-free promise broken."
    )
