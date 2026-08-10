"""Tool definitions and executor for chat function calling.

Coarse-grained tools tuned for a slow local model: batch reads, pruned
listings, regex search, and fuzzy path recovery — every wasted tool round
costs a full LLM inference (~1-2 min at local speeds).
"""

import difflib
import json
import os
import re
from pathlib import Path

# Directories that are never worth exploring from chat.
JUNK_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    ".gradle",
    "Pods",
    "DerivedData",
    ".Trash",
    "Library",
}

_FILE_CHAR_BUDGET = 12_000  # per file
_BATCH_CHAR_BUDGET = 48_000  # per read_files call
_PREVIEW_LINES = 30

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": (
                "Read one or MANY files at once. Always batch: request every file "
                "you expect to need in a single call. Large files are truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute paths of the files to read",
                    }
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search file CONTENTS with a regex (like grep -rn). Returns "
                "path:line matches. Use this to locate code instead of reading "
                "files one by one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "root": {"type": "string", "description": "Directory to search in"},
                    "glob": {
                        "type": "string",
                        "description": "Optional filename filter, e.g. *.py",
                    },
                },
                "required": ["pattern", "root"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List a directory as a small tree (2 levels deep). Junk dirs "
                "(node_modules, .git…) are pruned. If a README or manifest "
                "exists, a preview is included automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path to list"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by name pattern (**/*.py, **/*.{ts,tsx}, etc)",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                    "root": {"type": "string", "description": "Root directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Semantic search over the user's personal notes (Obsidian vault). "
                "Use for questions about the user's own projects, ideas, people or plans."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "k": {"type": "integer", "description": "Max results (default 6)"},
                },
                "required": ["query"],
            },
        },
    },
]


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _is_junk(path: Path) -> bool:
    return any(part in JUNK_DIRS for part in path.parts)


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    kept = text[:budget]
    total_lines = text.count("\n") + 1
    kept_lines = kept.count("\n") + 1
    return kept + f"\n… [truncated: showing {kept_lines} of {total_lines} lines]"


def _walk(root: Path, max_depth: int = 3, max_files: int = 4000):
    """os.walk with junk pruning and a depth/file cap."""
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if d not in JUNK_DIRS and not d.startswith("."))
        if rel_depth >= max_depth:
            dirnames[:] = []
        for fn in sorted(filenames):
            yield Path(dirpath) / fn
            seen += 1
            if seen >= max_files:
                return


def _closest_matches(missing: Path, n: int = 3) -> list[Path]:
    """Find real files whose name resembles a hallucinated path."""
    anchor = missing.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if not anchor.is_dir():
        return []
    by_name: dict[str, list[Path]] = {}
    for f in _walk(anchor):
        by_name.setdefault(f.name, []).append(f)
    close = difflib.get_close_matches(missing.name, by_name.keys(), n=n, cutoff=0.6)
    return [p for name in close for p in by_name[name]][:n]


def _read_one(path: Path, budget: int) -> str:
    if path.exists() and path.is_file():
        try:
            content = _truncate(path.read_text(encoding="utf-8", errors="replace"), budget)
        except OSError as e:
            return f"Error reading {path}: {e}"
        lang = path.suffix.lstrip(".")
        return f"### {path}\n```{lang}\n{content}\n```"
    if path.is_dir():
        return f"Error: {path} is a directory — use list_dir"
    # Hallucinated path: recover instead of burning a round on "not found".
    candidates = _closest_matches(path)
    if not candidates:
        return f"Error: file not found: {path} (no similar file nearby)"
    best = candidates[0]
    note = f"NOTE: {path} does not exist; reading closest match instead.\n"
    if len(candidates) > 1:
        others = ", ".join(str(c) for c in candidates[1:])
        note += f"Other candidates: {others}\n"
    return note + _read_one(best, budget)


def _preview_special_files(path: Path) -> str:
    """Head of README + manifest, so a list_dir answers 'what is this?' too."""
    out = []
    manifests = {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"}
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        if f.name.lower().startswith("readme") or f.name in manifests:
            try:
                head = "\n".join(
                    f.read_text(encoding="utf-8", errors="replace").splitlines()[:_PREVIEW_LINES]
                )
            except OSError:
                continue
            out.append(f"--- {f.name} (first {_PREVIEW_LINES} lines) ---\n{head}")
    return "\n\n".join(out)


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def _tree(path: Path, depth: int = 2, per_dir: int = 40, indent: str = "") -> list[str]:
    lines = []
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    except OSError as e:
        return [f"{indent}[error: {e}]"]
    for entry in entries[:per_dir]:
        if entry.name.startswith(".") and entry.name not in (".env.example",):
            continue
        if entry.is_dir():
            if entry.name in JUNK_DIRS:
                lines.append(f"{indent}{entry.name}/ (ignored)")
            else:
                lines.append(f"{indent}{entry.name}/")
                if depth > 1:
                    lines += _tree(entry, depth - 1, per_dir, indent + "  ")
        else:
            try:
                size = _fmt_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"{indent}{entry.name}  ({size})")
    if len(entries) > per_dir:
        lines.append(f"{indent}… and {len(entries) - per_dir} more entries")
    return lines


async def execute_tool(tool_call: dict, *, notes_search=None) -> str:
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        return f"Error: invalid arguments JSON — {e}"

    if name in ("read_files", "read_file"):
        paths = args.get("paths") or ([args["path"]] if args.get("path") else [])
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            return "Error: no paths given"
        budget = max(2000, min(_FILE_CHAR_BUDGET, _BATCH_CHAR_BUDGET // len(paths)))
        return "\n\n".join(_read_one(_resolve(p), budget) for p in paths)

    if name == "search":
        return _do_search(args)

    if name == "write_file":
        return "Error: write_file is disabled; Hearthia chat is read-only"

    if name == "list_dir":
        path = _resolve(args["path"])
        if not path.exists():
            candidates = _closest_matches(path / "x")  # anchor search at nearest ancestor
            hint = ""
            if candidates:
                dirs = sorted({str(c.parent) for c in candidates})
                hint = f"\nNearby existing dirs: {', '.join(dirs)}"
            return f"Error: path not found: {path}{hint}"
        if not path.is_dir():
            return f"Error: not a directory: {path} — use read_files"
        body = "\n".join(_tree(path)[:200])
        preview = _preview_special_files(path)
        out = f"📁 {path}\n{body}"
        if preview:
            out += f"\n\n{preview}"
        return out

    if name == "glob":
        root = _resolve(args.get("root", "~"))
        pattern = args["pattern"]
        if not root.exists():
            return f"Error: root not found: {root}"
        try:
            hits = []
            for f in root.rglob(pattern):
                if f.is_file() and not _is_junk(f.relative_to(root)):
                    hits.append(str(f.relative_to(root)))
                    if len(hits) >= 500:
                        break
        except (OSError, ValueError) as e:
            return f"Error searching files: {e}"
        if not hits:
            return f"No files matching '{pattern}' in {root}"
        shown = "\n".join(sorted(hits)[:200])
        more = f"\n… and {len(hits) - 200} more" if len(hits) > 200 else ""
        return f"{len(hits)} files matching '{pattern}' in {root}\n{shown}{more}"

    if name == "search_notes":
        if notes_search is None:
            return "Error: notes search not available (brain vault not configured)"
        try:
            return await notes_search(args["query"], int(args.get("k", 6)))
        except Exception as e:
            return f"Error searching notes: {e}"

    return f"Error: unknown tool '{name}'"


def _do_search(args: dict) -> str:
    root = _resolve(args["root"])
    if not root.is_dir():
        return f"Error: root not found or not a directory: {root}"
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        return f"Error: bad regex — {e}"
    name_filter = args.get("glob")
    matches: list[str] = []
    for f in _walk(root, max_depth=6, max_files=5000):
        if name_filter and not f.match(name_filter):
            continue
        try:
            if f.stat().st_size > 512_000:
                continue
            blob = f.read_bytes()
            if b"\0" in blob[:1024]:
                continue
            text = blob.decode("utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = f.relative_to(root)
                matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(matches) >= 100:
                    break
        if len(matches) >= 100:
            break
    if not matches:
        return f"No matches for /{args['pattern']}/ in {root}"
    return f"{len(matches)} matches for /{args['pattern']}/ in {root}\n" + "\n".join(matches)
