"""GGUF weight dedupe across runtimes.

Ollama, LM Studio and Hearthia's own ``models/`` folder each keep an
independent copy of any GGUF a user has downloaded more than once for
different tools — byte-identical weights, just reachable under different
names and paths. No local-model runtime looks across those folders for
exact duplicates and offers to reclaim the disk; every extra client
silently doubles it.

Detection groups by file size first (a cheap ``stat()``), then hashes only
within same-size groups — hashing every multi-gigabyte GGUF up front would
make a routine `hearth dedupe` impractically slow.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hearthia.dedupe")

_HASH_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class DuplicateGroup:
    sha256: str
    size: int
    paths: tuple[Path, ...]

    @property
    def wasted_bytes(self) -> int:
        return self.size * (len(self.paths) - 1)


def _hash_file(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK_BYTES):
                h.update(chunk)
    except OSError as e:
        log.debug("could not hash %s: %s", path, e)
        return None
    return h.hexdigest()


def find_gguf_files(roots: list[Path]) -> list[Path]:
    """Every ``.gguf`` file under the given roots, each real path listed once."""
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.gguf")):
            if not path.is_file():
                continue
            try:
                real = path.resolve()
            except OSError:
                continue
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return out


def find_duplicates(paths: list[Path]) -> list[DuplicateGroup]:
    """Group byte-identical files: same size, then same content hash."""
    by_size: dict[int, list[Path]] = {}
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        by_size.setdefault(size, []).append(path)

    groups: list[DuplicateGroup] = []
    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue
        by_hash: dict[str, list[Path]] = {}
        for path in candidates:
            digest = _hash_file(path)
            if digest:
                by_hash.setdefault(digest, []).append(path)
        for digest, dup_paths in by_hash.items():
            if len(dup_paths) >= 2:
                groups.append(DuplicateGroup(digest, size, tuple(dup_paths)))
    return groups


def default_roots() -> list[Path]:
    """The usual local-model weight folders: Hearthia, Ollama, LM Studio."""
    return [
        Path.home() / ".hearthia" / "models",
        Path.home() / ".ollama",
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ]


def link_duplicates(group: DuplicateGroup) -> tuple[list[Path], list[str]]:
    """Replace every path but the first with a hardlink to it.

    Same-filesystem only: a cross-volume group cannot be hardlinked and is
    reported as a failure, never silently copied. Returns (paths relinked,
    error messages for paths that could not be).
    """
    keep, *rest = group.paths
    relinked: list[Path] = []
    errors: list[str] = []
    for path in rest:
        tmp = path.with_name(path.name + ".dedupe-tmp")
        try:
            if tmp.exists():
                tmp.unlink()
            os.link(keep, tmp)
            os.replace(tmp, path)
            relinked.append(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return relinked, errors
