"""Command-line entry point for coreml_diffusion.

Mirrors ``coreml_diffusion.convert`` so the package can produce a Core ML
``.mlpackage`` with no ComfyUI involved — for headless and on-device (iOS)
conversion workflows. The heavy import (coremltools/diffusers, pulled by
``convert``) is deferred into the handler, so ``--help`` and argument parsing
stay light and the arg→call mapping is testable on plain Linux.

Example:
    coreml-diffusion convert --ckpt model.safetensors --model-version SD15 \\
        --out unet.mlpackage --height 512 --width 512 --attn-impl SPLIT_EINSUM
"""

import argparse

import coreml_diffusion


def _parse_lora(spec):
    """Parse a ``PATH:STRENGTH`` lora spec into ``(path, float_strength)``.

    Strength defaults to 1.0 when omitted. ``rsplit`` on the last ':' so Windows
    drive letters / colons in the path survive.
    """
    path, sep, strength = spec.rpartition(":")
    if not sep:
        return spec, 1.0
    return path, float(strength)


def _convert_cmd(args):
    sample_size = (args.height // 8, args.width // 8)
    lora_weights = [_parse_lora(spec) for spec in (args.lora or [])]
    coreml_diffusion.convert(
        args.ckpt,
        coreml_diffusion.ModelVersion[args.model_version],
        args.out,
        batch_size=args.batch_size,
        sample_size=sample_size,
        controlnet_support=args.controlnet,
        lora_weights=lora_weights or None,
        attn_impl=args.attn_impl,
        config_path=args.config,
        quantize_nbits=args.quantize,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="coreml-diffusion",
        description="Convert diffusion checkpoints to Core ML for Apple Neural Engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="Convert a checkpoint's UNet to a .mlpackage")
    conv.add_argument(
        "--ckpt", required=True, help="Path to the source .safetensors checkpoint"
    )
    conv.add_argument(
        "--model-version",
        required=True,
        # include experimental: the CLI is the power-user path. Experimental
        # versions (LCM, SDXL_REFINER) convert but are not golden-verified.
        choices=coreml_diffusion.list_model_versions(include_experimental=True),
        help="Model architecture (verified: SD15, SDXL; experimental otherwise)",
    )
    conv.add_argument("--out", required=True, help="Output .mlpackage path to write")
    conv.add_argument(
        "--height", type=int, default=512, help="Target image height (default 512)"
    )
    conv.add_argument(
        "--width", type=int, default=512, help="Target image width (default 512)"
    )
    conv.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default 1)"
    )
    conv.add_argument(
        "--attn-impl",
        choices=coreml_diffusion.list_attention_impls(),
        default=coreml_diffusion.list_attention_impls()[0],
        help="Attention implementation (default SPLIT_EINSUM)",
    )
    conv.add_argument(
        "--controlnet",
        action="store_true",
        help="Add ControlNet residual inputs to the converted UNet",
    )
    conv.add_argument(
        "--lora",
        action="append",
        metavar="PATH[:STRENGTH]",
        help="LoRA to fuse before conversion; repeatable. STRENGTH defaults to 1.0",
    )
    conv.add_argument(
        "--config", default=None, help="Optional original-config YAML path"
    )
    conv.add_argument(
        "--quantize",
        choices=coreml_diffusion.list_quant_modes(),
        default="none",
        help="K-means weight palettization bits (default none = unquantized)",
    )
    conv.set_defaults(func=_convert_cmd)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
