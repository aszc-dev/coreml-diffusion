"""Tier 0: model source registry + checkpoint name resolution (framework-free)."""

import pytest

from coreml_diffusion import sources


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point the registry at a throwaway TOML so tests never touch user config."""
    cfg = tmp_path / "sources.toml"
    monkeypatch.setenv("COREML_DIFFUSION_CONFIG", str(cfg))
    return cfg


def _comfy_tree(base, *ckpt_stems):
    """Build a minimal comfy-layout source dir with the given checkpoint stems."""
    ckpt_dir = base / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for stem in ckpt_stems:
        (ckpt_dir / f"{stem}.safetensors").write_text("")
    return base


def test_add_then_resolve_by_name(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "v1-5-pruned-emaonly")
    sources.add_source("comfy", str(tmp_path / "comfy"), kind="comfy")

    resolved = sources.resolve_checkpoint("v1-5-pruned-emaonly")
    assert resolved.endswith("checkpoints/v1-5-pruned-emaonly.safetensors")


def test_resolve_accepts_name_with_extension(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "model")
    sources.add_source("comfy", str(tmp_path / "comfy"))
    assert sources.resolve_checkpoint("model.safetensors").endswith("model.safetensors")


def test_existing_path_passes_through(isolated_config, tmp_path):
    f = tmp_path / "explicit.safetensors"
    f.write_text("")
    assert sources.resolve_checkpoint(str(f)) == str(f.resolve())


def test_registry_round_trip_and_persistence(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "a", "b")
    sources.add_source("comfy", str(tmp_path / "comfy"))
    # Re-read from disk (fresh call) to prove TOML write/read round-trips.
    loaded = sources.load_sources()
    assert loaded["comfy"]["kind"] == "comfy"
    assert sources.iter_checkpoints(loaded["comfy"]) == ["a", "b"]


def test_flat_kind_reads_checkpoints_at_root(isolated_config, tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "loose.safetensors").write_text("")
    sources.add_source("flat", str(flat), kind="flat")
    assert sources.resolve_checkpoint("loose").endswith("flat/loose.safetensors")


def test_missing_checkpoint_raises(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "present")
    sources.add_source("comfy", str(tmp_path / "comfy"))
    with pytest.raises(FileNotFoundError):
        sources.resolve_checkpoint("absent")


def test_ambiguous_match_raises(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "a", "dup")
    _comfy_tree(tmp_path / "b", "dup")
    sources.add_source("a", str(tmp_path / "a"))
    sources.add_source("b", str(tmp_path / "b"))
    with pytest.raises(ValueError, match="ambiguous"):
        sources.resolve_checkpoint("dup")
    # --source disambiguates.
    assert sources.resolve_checkpoint("dup", source="b").endswith(
        "/b/checkpoints/dup.safetensors"
    )


def test_unknown_source_name_raises(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "x")
    sources.add_source("comfy", str(tmp_path / "comfy"))
    with pytest.raises(KeyError):
        sources.resolve_checkpoint("x", source="nope")


def test_add_rejects_bad_kind_and_missing_dir(isolated_config, tmp_path):
    (tmp_path / "real").mkdir()
    with pytest.raises(ValueError):
        sources.add_source("x", str(tmp_path / "real"), kind="bogus")
    with pytest.raises(NotADirectoryError):
        sources.add_source("x", str(tmp_path / "does-not-exist"))


def test_remove_source(isolated_config, tmp_path):
    _comfy_tree(tmp_path / "comfy", "x")
    sources.add_source("comfy", str(tmp_path / "comfy"))
    sources.remove_source("comfy")
    assert sources.load_sources() == {}
    with pytest.raises(KeyError):
        sources.remove_source("comfy")
