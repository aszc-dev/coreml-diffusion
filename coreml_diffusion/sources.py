"""Model source registry — resolve checkpoint names against known directories.

A *source* is a base directory holding model files under a known layout, so the
CLI can take ``--ckpt v1-5-pruned-emaonly`` instead of a full path. The first
supported layout is ComfyUI's ``models/`` tree (``kind="comfy"``); ``kind="flat"``
is a plain directory of checkpoints. More source kinds are expected.

Framework-free: this only reads directories and a TOML config — it never imports
``comfy``/``folder_paths``. The registry is persisted as user-global TOML at
``$XDG_CONFIG_HOME/coreml-diffusion/sources.toml`` (override with
``COREML_DIFFUSION_CONFIG``). The library ``convert`` stays path-only; name
resolution is a CLI-side convenience layered on top.
"""

import os
import tomllib
from pathlib import Path

# Subdirectory layout per source kind. Keyed by a logical category so callers ask
# for "checkpoints" without knowing the on-disk folder name.
_KIND_LAYOUT = {
    "comfy": {
        "checkpoints": "checkpoints",
        "loras": "loras",
        "vae": "vae",
        "controlnet": "controlnet",
        "configs": "configs",
    },
    "flat": {"checkpoints": "."},
}

SOURCE_KINDS = tuple(_KIND_LAYOUT)

# Checkpoint file extensions, in resolution-preference order.
_CKPT_EXTS = (".safetensors", ".ckpt")


def config_path() -> Path:
    """Path to the sources registry TOML (honoring the env overrides)."""
    override = os.environ.get("COREML_DIFFUSION_CONFIG")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "coreml-diffusion" / "sources.toml"


def load_sources() -> dict[str, dict[str, str]]:
    """Registered sources as ``{name: {"path": ..., "kind": ...}}`` (empty if none)."""
    path = config_path()
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("sources", {})


def _dump_sources(sources: dict[str, dict[str, str]]) -> str:
    """Serialize the registry to TOML. Flat, tool-managed schema — no nesting."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = []
    for name, entry in sorted(sources.items()):
        lines.append(f"[sources.{name}]")
        lines.append(f'path = "{esc(entry["path"])}"')
        lines.append(f'kind = "{esc(entry["kind"])}"')
        lines.append("")
    return "\n".join(lines)


def save_sources(sources: dict[str, dict[str, str]]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_sources(sources))


def add_source(name: str, path: str, kind: str = "comfy") -> dict[str, str]:
    """Register (or overwrite) a source. Validates kind and that the dir exists."""
    if kind not in _KIND_LAYOUT:
        raise ValueError(
            f"Unknown source kind {kind!r}; expected one of {SOURCE_KINDS}."
        )
    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {resolved}")
    sources = load_sources()
    sources[name] = {"path": resolved, "kind": kind}
    save_sources(sources)
    return sources[name]


def remove_source(name: str) -> None:
    sources = load_sources()
    if name not in sources:
        raise KeyError(f"No source named {name!r}.")
    del sources[name]
    save_sources(sources)


def category_dir(entry: dict[str, str], category: str) -> Path:
    """Resolve a source entry's directory for a logical category (e.g. checkpoints)."""
    subdir = _KIND_LAYOUT[entry["kind"]][category]
    return Path(entry["path"]) / subdir


def iter_checkpoints(entry: dict[str, str]) -> list[str]:
    """Checkpoint file stems available under a source (sorted, deduped)."""
    ckpt_dir = category_dir(entry, "checkpoints")
    if not ckpt_dir.is_dir():
        return []
    stems = {
        p.stem for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in _CKPT_EXTS
    }
    return sorted(stems)


def resolve_checkpoint(name_or_path: str, source: str | None = None) -> str:
    """Resolve a checkpoint name (or path) to an absolute file path.

    An existing path is returned as-is. Otherwise ``name_or_path`` is treated as a
    checkpoint stem (with or without extension) and searched in the checkpoints
    directory of the named source, or of every source when ``source`` is None.
    Raises on no match, or on an ambiguous match across sources.
    """
    direct = Path(name_or_path).expanduser()
    if direct.exists():
        return str(direct.resolve())

    sources = load_sources()
    if source is not None:
        if source not in sources:
            raise KeyError(f"No source named {source!r}. Known: {sorted(sources)}.")
        sources = {source: sources[source]}

    # Resolution is an opt-in convenience layer. Defer to the caller (which does
    # its own existence check) for path-like inputs or when no sources exist, so
    # the explicit-path workflow and the no-config default behave as before.
    looks_like_path = os.sep in name_or_path or (
        os.altsep is not None and os.altsep in name_or_path
    )
    if looks_like_path or not sources:
        return name_or_path

    stem = name_or_path
    for ext in _CKPT_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    matches = []
    for src_name, entry in sources.items():
        ckpt_dir = category_dir(entry, "checkpoints")
        for ext in _CKPT_EXTS:
            candidate = ckpt_dir / f"{stem}{ext}"
            if candidate.is_file():
                matches.append((src_name, str(candidate.resolve())))
                break

    if not matches:
        raise FileNotFoundError(
            f"Checkpoint {name_or_path!r} not found in sources {sorted(sources)}."
        )
    if len(matches) > 1:
        where = ", ".join(f"{s}:{p}" for s, p in matches)
        raise ValueError(
            f"Checkpoint {name_or_path!r} is ambiguous across sources ({where}). "
            f"Disambiguate with --source NAME."
        )
    return matches[0][1]
