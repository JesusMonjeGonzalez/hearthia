"""Explicit project context, generated in a disposable filesystem worker."""

import hashlib
from pathlib import Path

from hearthia.api.repomap import build_repo_map, detect_paths
from hearthia.api.tools import _read_prefix


def project_context(workspace: str, text: str, *, mode: str = "read") -> dict:
    roots = [workspace] if workspace else detect_paths(text)[:2]
    blocks, paths = [], []
    for value in roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            if workspace:
                raise ValueError(f"Workspace is not a directory: {root}")
            continue
        paths.append(str(root))
        blocks.append(build_repo_map(root))
        # Only the explicitly selected workspace supplies project instructions.
        # Mentioning a directory in prose does not grant it instruction authority.
        if workspace:
            blocks.append(
                f"Working directory: {root}. Resolve relative tool paths here. "
                f"Mode: {mode}. Check deeper AGENTS.md files when working in their directories."
            )
            instructions = root / "AGENTS.md"
            if instructions.is_file():
                blocks.append(
                    f"Project instructions ({instructions}):\n" + _read_prefix(instructions, 8000)
                )
    content = "\n\n".join(blocks)
    # Deterministic signature: identical workspace state -> identical bytes,
    # which is what lets the chat pin this block and keep the prompt prefix
    # cached instead of re-prefilling the repo map on every turn.
    signature = hashlib.sha256(content.encode()).hexdigest()[:16]
    return {"content": content, "roots": paths, "signature": signature}
