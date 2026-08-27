"""Adopt models you already have: Ollama blobs, LM Studio folders, plain dirs.

Nobody should re-download 20 GB because they switched runtimes. This module
finds GGUF weights that are already on disk, reads their headers, and turns
them into Hearthia config blocks.
"""

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hearthia.gguf import model_ram_profile
from hearthia.library import estimate_resident_ram, kv_cache_bytes

log = logging.getLogger("hearthia.adopt")

# reference context for estimates when the file's own training context is huge
_REFERENCE_CTX = 32768


@dataclass(frozen=True)
class AdoptedModel:
    name: str
    path: Path
    size_bytes: int
    est_resident_bytes: int
    known: bool  # estimate derived from a real GGUF header


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _estimate(path: Path) -> tuple[int, bool]:
    profile = model_ram_profile(path)
    if profile is None:
        size = path.stat().st_size
        return int(size * 1.3), False
    ctx = min(profile.context_length, _REFERENCE_CTX)
    kv = kv_cache_bytes(profile.n_layer, profile.n_kv_heads, profile.k_len, profile.v_len, ctx)
    return estimate_resident_ram(profile.file_size, kv), True


def scan_dir(root: Path) -> list[AdoptedModel]:
    """Find .gguf files under root (any depth), largest last."""
    out: list[AdoptedModel] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*.gguf")):
        if path.suffix == ".gguf" and path.is_file():
            out.append(_adopted(path.stem, path))
    return out


def _adopted(name: str, path: Path) -> AdoptedModel:
    size = path.stat().st_size
    est, known = _estimate(path)
    return AdoptedModel(
        name=_slug(name), path=path, size_bytes=size, est_resident_bytes=est, known=known
    )


def iter_ollama_manifests(ollama_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (model name, blob path) for every Ollama manifest on disk.

    Ollama stores weights as content-addressed blobs referenced by JSON
    manifests; the GGUF is the layer with the image.model media type.
    """
    manifests = ollama_dir / "manifests"
    if not manifests.exists():
        return
    for manifest in sorted(manifests.rglob("*")):
        if not manifest.is_file():
            continue
        try:
            doc = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        # relative name: registry.ollama.ai/library/qwen3:8b -> qwen3:8b
        rel = manifest.relative_to(manifests).with_suffix("")
        parts = [p for p in rel.parts if p not in ("library", "registry.ollama.ai")]
        name = "/".join(parts[:-1]) + ":" + parts[-1] if len(parts) > 1 else manifest.stem
        for layer in doc.get("layers") or []:
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = str(layer.get("digest", ""))
                if digest.startswith("sha256:"):
                    blob = ollama_dir / "blobs" / f"sha256-{digest.split(':', 1)[1]}"
                    if blob.exists():
                        yield name, blob


def scan_ollama(ollama_dir: Path) -> list[AdoptedModel]:
    """Every GGUF Ollama has already pulled, under its real model name."""
    out: list[AdoptedModel] = []
    for name, blob in iter_ollama_manifests(ollama_dir):
        out.append(_adopted(name.split(":")[0] + "-" + name.split(":")[-1], blob))
    return out


# Where to look by default, newest runtime layout first.
_LMSTUDIO_DIRS = (
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / ".ollama" / "models",
)


def default_candidates() -> list[tuple[str, list[AdoptedModel]]]:
    """Probe the usual runtimes so `hearth scan` shows something useful."""
    found: list[tuple[str, list[AdoptedModel]]] = []
    ollama = Path.home() / ".ollama"
    if (ollama / "manifests").exists():
        models = scan_ollama(ollama)
        if models:
            found.append(("ollama", models))
    for d in _LMSTUDIO_DIRS:
        models = scan_dir(d)
        if models:
            found.append((str(d), models))
            break
    return found
