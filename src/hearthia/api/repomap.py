"""Compact project maps injected into chat context.

A slow local model burns 1-2 minutes per tool round; handing it a pruned
tree + README/manifest previews up front removes the 2-3 discovery rounds
(list_dir, read README…) it would otherwise spend on every conversation.
"""

import re
from pathlib import Path

from hearthia.api.tools import JUNK_DIRS, _fmt_size

_DEFAULT_BUDGET = 4000
_PREVIEW_LINES = 30
_PREVIEW_CHARS = 1200
_MANIFESTS = {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"}

# root str -> (signature, rendered map)
_cache: dict[str, tuple[tuple, str]] = {}

_PATH_RE = re.compile(r"(?:^|[\s(¿¡])((?:~|/[\w.@-]+)(?:/[\w.@-]+)+/?)")
_QUOTED_PATH_RE = re.compile(r"(['\"])((?:~|/)[^'\"\r\n]+)\1")
_PATH_END_RE = re.compile(r"[,;:!?\n\r\t<>|]")


def _clean_path(raw: str) -> str:
    return raw.rstrip(".,;:!?").rstrip("/")


def _longest_existing_path(text: str, match: re.Match[str]) -> str:
    """Extend an unquoted path through spaces only when that prefix exists."""
    stop = _PATH_END_RE.search(text, match.end(1))
    end = stop.start() if stop else len(text)
    words = text[match.start(1) : end].split()
    for count in range(len(words), 1, -1):
        candidate = _clean_path(" ".join(words[:count]))
        if Path(candidate).expanduser().exists():
            return candidate
    return _clean_path(match.group(1))


def detect_paths(text: str) -> list[str]:
    """Filesystem paths mentioned in a message (absolute or ~/), deduped."""
    found: list[str] = []
    quoted_ranges: list[tuple[int, int]] = []
    for match in _QUOTED_PATH_RE.finditer(text):
        quoted_ranges.append(match.span())
        expanded = str(Path(_clean_path(match.group(2))).expanduser())
        if expanded not in found:
            found.append(expanded)
    for m in _PATH_RE.finditer(text):
        if any(start <= m.start(1) < end for start, end in quoted_ranges):
            continue
        raw = _longest_existing_path(text, m)
        expanded = str(Path(raw).expanduser())
        if expanded not in found:
            found.append(expanded)
    return found


def _signature(root: Path) -> tuple:
    try:
        parts = [root.stat().st_mtime]
        parts += [e.stat().st_mtime for e in sorted(root.iterdir())[:50]]
        return tuple(parts)
    except OSError:
        return ()


def _tree_lines(path: Path, depth: int, indent: str = "") -> list[str]:
    lines: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    except OSError:
        return lines
    for entry in entries[:60]:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name in JUNK_DIRS:
                continue
            lines.append(f"{indent}{entry.name}/")
            if depth > 1:
                lines += _tree_lines(entry, depth - 1, indent + "  ")
        else:
            try:
                size = _fmt_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"{indent}{entry.name} ({size})")
    if len(entries) > 60:
        lines.append(f"{indent}… {len(entries) - 60} more entries")
    return lines


def _previews(root: Path) -> str:
    out = []
    try:
        files = sorted(p for p in root.iterdir() if p.is_file())
    except OSError:
        return ""
    for f in files:
        if f.name.lower().startswith("readme") or f.name in _MANIFESTS:
            try:
                head = "\n".join(
                    f.read_text(encoding="utf-8", errors="replace").splitlines()[:_PREVIEW_LINES]
                )[:_PREVIEW_CHARS]
            except OSError:
                continue
            out.append(f"--- {f.name} ---\n{head}")
    return "\n\n".join(out)


def build_repo_map(root: Path, budget: int = _DEFAULT_BUDGET) -> str:
    root = root.expanduser().resolve()
    sig = _signature(root)
    cached = _cache.get(str(root))
    if cached and cached[0] == sig:
        return cached[1]

    previews = _previews(root)[: budget // 2]
    remaining = budget - len(previews)
    tree = []
    used = 0
    for line in _tree_lines(root, depth=3):
        used += len(line) + 1
        if used > remaining:
            tree.append("… (tree truncated)")
            break
        tree.append(line)

    out = f"Project map of {root} (auto-generated):\n" + "\n".join(tree)
    if previews:
        out += f"\n\n{previews}"
    _cache[str(root)] = (sig, out)
    return out
