"""Tier-0 arg→call mapping for the coreml-diffusion CLI.

The heavy ``coreml_diffusion.convert`` is mocked, so this runs on plain Linux
and never imports coremltools/diffusers — only that the parsed flags reach
``convert`` with the right shapes/types.
"""

import pytest

import coreml_diffusion
from coreml_diffusion import ModelVersion, cli


@pytest.fixture(autouse=True)
def _isolate_sources(tmp_path, monkeypatch):
    """Point the source registry at an empty throwaway config so arg-mapping
    tests never read the developer's real ~/.config sources (which would resolve
    or reject bare --ckpt names and make these tests non-hermetic)."""
    monkeypatch.setenv("COREML_DIFFUSION_CONFIG", str(tmp_path / "sources.toml"))


def _run(monkeypatch, argv):
    captured = {}

    def fake_convert(ckpt, model_version, out, **kwargs):
        captured["positional"] = (ckpt, model_version, out)
        captured["kwargs"] = kwargs

    # Set the module __dict__ entry directly. monkeypatch.setattr would first
    # getattr(coreml_diffusion, "convert"), which fires the lazy __getattr__ and
    # imports the real (heavy) convert module — breaking Tier-0 purity. setitem
    # bypasses that; the real dict entry then shadows __getattr__.
    monkeypatch.setitem(coreml_diffusion.__dict__, "convert", fake_convert)
    cli.main(argv)
    return captured


def test_full_args_map_to_convert(monkeypatch):
    captured = _run(
        monkeypatch,
        [
            "convert",
            "--ckpt",
            "model.safetensors",
            "--model-version",
            "SD15",
            "--out",
            "unet.mlpackage",
            "--height",
            "768",
            "--width",
            "512",
            "--batch-size",
            "2",
            "--attn-impl",
            "ORIGINAL",
            "--controlnet",
            "--quantize",
            "4",
            "--lora",
            "a.safetensors:0.8",
            "--lora",
            "b.safetensors",
        ],
    )
    ckpt, model_version, out = captured["positional"]
    assert ckpt == "model.safetensors"
    assert model_version is ModelVersion.SD15
    assert out == "unet.mlpackage"

    kw = captured["kwargs"]
    assert kw["sample_size"] == (768 // 8, 512 // 8)
    assert kw["batch_size"] == 2
    assert kw["attn_impl"] == "ORIGINAL"
    assert kw["controlnet_support"] is True
    assert kw["quantize_nbits"] == "4"
    assert kw["lora_weights"] == [("a.safetensors", 0.8), ("b.safetensors", 1.0)]


def test_defaults(monkeypatch):
    captured = _run(
        monkeypatch,
        [
            "convert",
            "--ckpt",
            "m.safetensors",
            "--model-version",
            "SDXL",
            "--out",
            "o.mlpackage",
        ],
    )
    kw = captured["kwargs"]
    assert captured["positional"][1] is ModelVersion.SDXL
    assert kw["sample_size"] == (64, 64)  # 512 // 8
    assert kw["batch_size"] == 1
    assert kw["attn_impl"] == "SPLIT_EINSUM"
    assert kw["controlnet_support"] is False
    assert kw["quantize_nbits"] == "none"
    assert kw["lora_weights"] is None  # no --lora -> None, not []
    assert kw["component"] == "unet"  # default component


def test_component_flag_maps_to_convert(monkeypatch):
    captured = _run(
        monkeypatch,
        [
            "convert",
            "--ckpt",
            "m.safetensors",
            "--model-version",
            "SDXL",
            "--out",
            "te2.mlpackage",
            "--component",
            "text_encoder_2",
        ],
    )
    assert captured["kwargs"]["component"] == "text_encoder_2"


def test_bad_component_rejected():
    with pytest.raises(SystemExit):
        cli.main(
            [
                "convert",
                "--ckpt",
                "m",
                "--model-version",
                "SD15",
                "--out",
                "o",
                "--component",
                "refiner",  # not a convertible component
            ]
        )


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_missing_required_args_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["convert", "--ckpt", "m.safetensors"])  # no --model-version/--out
    assert exc.value.code != 0


def test_bad_model_version_rejected():
    with pytest.raises(SystemExit):
        cli.main(
            ["convert", "--ckpt", "m", "--model-version", "sd15", "--out", "o"]
        )  # lowercase != .name
