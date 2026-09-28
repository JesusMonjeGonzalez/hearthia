"""Opt-in coding capabilities: exact edits, new files and bounded commands.

The file tools constrain writes to a selected workspace. Command execution is
not a sandbox: it runs with the local user's permissions in that directory.
"""

import ast
import difflib
import hashlib
import json
import os
import stat
import tempfile
import tomllib
from pathlib import Path

MAX_FILE_BYTES = 512_000
MAX_DIFF_CHARS = 20_000
MUTATING_TOOLS = {"edit_file", "create_file", "run_command", "update_plan", "task", "job"}


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


CODING_TOOLS = [
    _schema(
        "edit_file",
        "Replace one exact, unique text block in an existing workspace file. "
        "Read it first. Returns a unified diff. Ambiguous/stale matches are refused.",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "expected_sha256": {
                "type": "string",
                "description": "Optional hash from a previous read",
            },
        },
        ["path", "old_text", "new_text"],
    ),
    _schema(
        "create_file",
        "Create a new UTF-8 file in the workspace. Never overwrites an existing "
        "file. Parent directories must exist. Returns a unified diff.",
        {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["path", "content"],
    ),
    _schema(
        "task",
        "Delegate a focused, read-only investigation to a subagent with its own "
        "disposable context (it can read files, search and run commands, but not "
        "edit). Use it for exploration whose raw output you do not want in this "
        "conversation; it returns a bounded report with findings and evidence.",
        {
            "description": {"type": "string", "maxLength": 160},
            "prompt": {"type": "string", "maxLength": 4_000},
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 12},
        },
        ["description", "prompt"],
    ),
    _schema(
        "update_plan",
        "Record or refresh the step plan for this task so it survives context "
        "trimming and keeps long sessions on track. Call it when the plan changes "
        "and whenever a step finishes (send the full steps list plus the 1-based "
        "indices of finished steps in done); 1..24 short steps, max 160 chars.",
        {
            "steps": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "minItems": 1,
                "maxItems": 24,
            },
            "done": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 24},
                "description": "1-based indices of finished steps",
            },
        },
        ["steps"],
    ),
    _schema(
        "run_command",
        "Run an executable with an argument array in the workspace. "
        "No implicit shell; use project-native test/build commands. Returns exit_code, "
        "duration, bounded output and resource-limit status. Do not start background services.",
        {
            "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
            "cwd": {"type": "string", "description": "Optional workspace subdirectory"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            "background": {
                "type": "boolean",
                "description": "Run as a background job and return immediately; poll with job()",
            },
        },
        ["argv"],
    ),
    _schema(
        "job",
        "Manage background commands started with run_command(background=true): "
        "list them, read a bounded status+tail, wait (blocking, up to 120 s) or "
        "stop one. Job output lives in a log file; only a bounded tail reaches "
        "this conversation.",
        {
            "action": {"type": "string", "enum": ["list", "status", "wait", "stop"]},
            "id": {"type": "string", "description": "Job id (required except for list)"},
            "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
        },
        ["action"],
    ),
]


def _syntax_check(path: Path, text: str) -> str | None:
    """Cheap local parse of the edited file: Python, JSON, TOML.

    Costs nothing in tokens unless it fails, needs no subprocess and no
    inference. It catches the most common local-model edit error — a file
    that no longer parses — so the model can fix it in the same turn instead
    of spending a whole extra round discovering it. Never blocks the edit.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(text)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        else:
            return None
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}" if exc.lineno else str(exc)
    except (ValueError, TypeError) as exc:
        return str(exc)[:200]
    return None


def workspace_path(workspace: str, value: str = ".") -> Path:
    if not workspace:
        raise ValueError("Select a workspace before using coding tools")
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace must be a directory")
    path = Path(value).expanduser()
    lexical = Path(os.path.abspath(path if path.is_absolute() else root / path))
    resolved = lexical.resolve()
    if not lexical.is_relative_to(root) or not resolved.is_relative_to(root):
        raise ValueError("Path must stay inside the selected workspace")
    for part in lexical.relative_to(root).parts:
        if part == ".git":
            raise ValueError("Repository metadata cannot be edited by file tools")
    cursor = lexical
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("Coding tools do not follow symlink paths")
        cursor = cursor.parent
    return lexical


def _read_snapshot(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Edits require a regular file")
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 512 KB edit limit")
    if b"\0" in data:
        raise ValueError("Binary files cannot be edited")
    return data, info


def apply_edit(name: str, args: dict, workspace: str) -> dict:
    path_value = args.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("path must be a non-empty string")
    path = workspace_path(workspace, path_value)
    if not path.parent.is_dir():
        raise ValueError("Parent directory must already exist")
    before, info = b"", None
    if name == "create_file":
        text = args.get("content")
        if not isinstance(text, str):
            raise ValueError("content must be text")
        if path.exists():
            raise ValueError("File already exists; read it and use edit_file")
    elif name == "edit_file":
        before, info = _read_snapshot(path)
        original = before.decode("utf-8")
        expected = args.get("expected_sha256")
        if expected is not None and expected != hashlib.sha256(before).hexdigest():
            raise ValueError("File changed since the supplied hash; read it again")
        old, new = args.get("old_text"), args.get("new_text")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old_text must be non-empty text and new_text must be text")
        if original.count(old) != 1:
            raise ValueError("old_text must match exactly once; read a larger, current block")
        text = original.replace(old, new, 1)
    else:
        raise ValueError("Unknown editing tool")
    after = text.encode("utf-8")
    if len(after) > MAX_FILE_BYTES or b"\0" in after:
        raise ValueError("Result must be UTF-8 text without NUL bytes, at most 512 KB")

    # Build the diff before modifying anything. Bound pathological repeated-line
    # diffs separately: the worker's CPU deadline can stop this phase safely.
    relative = str(path.relative_to(Path(workspace).expanduser().resolve()))
    diff = "".join(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"a/{relative}" if info else "/dev/null",
            tofile=f"b/{relative}",
        )
    )
    fd, temporary = tempfile.mkstemp(prefix=".hearthia-edit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(after)
            target.flush()
            os.fchmod(target.fileno(), stat.S_IMODE(info.st_mode) if info else 0o644)
            os.fsync(target.fileno())
        workspace_path(workspace, path_value)  # revalidate before publication
        if info:
            current, current_info = _read_snapshot(path)
            if current != before or (current_info.st_dev, current_info.st_ino) != (
                info.st_dev,
                info.st_ino,
            ):
                raise ValueError("File changed while preparing the edit; read it again")
            os.replace(temporary, path)
        else:
            # Linking a staged file publishes it atomically and refuses races
            # with another creator; unlike replace(), it never overwrites.
            os.link(temporary, path)
        result = {
            "ok": True,
            "kind": "edit",
            "path": relative,
            "before_sha256": hashlib.sha256(before).hexdigest() if info else None,
            "after_sha256": hashlib.sha256(after).hexdigest(),
            "diff": diff[:MAX_DIFF_CHARS],
            "diff_truncated": len(diff) > MAX_DIFF_CHARS,
            "bytes": len(after),
            "changed": before != after or info is None,
        }
        syntax_error = _syntax_check(path, text)
        if syntax_error is not None:
            # Applied as written, but the model needs to know immediately:
            # this is the cheapest possible verification round.
            result["syntax_error"] = syntax_error
            result["note"] = (
                "the file was written but no longer parses; fix it before running anything"
            )
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)
