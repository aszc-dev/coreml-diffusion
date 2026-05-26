"""Pytest bootstrap for coreml-diffusion.

Auto-applies tier markers from the directory a test lives in, and (when a single
tier is requested via ``-m``) skips the other tiers at collection time so Tier 0
on Linux never imports the Mac-only stack (coremltools) that Tier 1/2 pull in.
"""
import pytest

_TIER_BY_DIR = {
    "tests/unit": "unit",
    "tests/smoke": "smoke",
    "tests/m2": "m2",
    "tests/inference": "inference",
}

_TIER_DIRS = {
    "unit": ("/tests/unit/",),
    "smoke": ("/tests/smoke/",),
    "m2": ("/tests/m2/",),
    "inference": ("/tests/inference/",),
}


def pytest_ignore_collect(collection_path, config):
    expr = config.option.markexpr
    if expr not in _TIER_DIRS:
        return None
    allowed = _TIER_DIRS[expr]
    rel = str(collection_path).replace("\\", "/")
    if "/tests/" not in rel:
        return None
    if rel.endswith("/tests"):
        return None
    if any(frag in rel + "/" for frag in allowed):
        return None
    return True


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        for fragment, marker in _TIER_BY_DIR.items():
            if f"/{fragment}/" in path:
                item.add_marker(getattr(pytest.mark, marker))
                break
