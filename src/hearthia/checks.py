"""Per-edit check commands: cheap, honest, configuration-driven.

Not an LSP: after a successful edit the configured command for that extension
runs once, with the same process-group cleanup, RSS sampling and wall deadline
as any other foreground command. A pass costs a few tokens; a failure carries
bounded output so the model can fix it in the same round instead of spending a
whole extra one. Nothing runs unless the user configured it.
"""

import json
import logging
from pathlib import Path

from hearthia.commands import run_command
from hearthia.settings import CheckSettings

log = logging.getLogger("hearthia.checks")

OUTPUT_CHARS = 2_000
_MAX_COMMAND_CHARS = 4_000


def find_check(checks: list[CheckSettings], relative_path: str) -> CheckSettings | None:
    suffix = Path(relative_path).suffix.lower().lstrip(".")
    if not suffix:
        return None
    for check in checks:
        if suffix in {ext.lower().lstrip(".") for ext in check.extensions}:
            return check
    return None


def _argv(check: CheckSettings, relative_path: str) -> list[str] | None:
    argv = [part.replace("{file}", relative_path) for part in check.command]
    if sum(len(part) for part in argv) > _MAX_COMMAND_CHARS:
        return None
    if not all(part and "\0" not in part for part in argv):
        return None
    return argv


async def run_check(
    checks: list[CheckSettings],
    workspace: str,
    relative_path: str,
    *,
    memory_limit: int,
) -> str | dict | None:
    """Run the matching check for this file, or None when nothing matches.

    Returns a compact string when the check passes and a bounded dict when it
    fails or cannot run — the caller embeds it in the edit tool result.
    """
    check = find_check(checks, relative_path)
    if check is None:
        return None
    argv = _argv(check, relative_path)
    if argv is None:
        return "Error: the configured check command is invalid"
    result = await run_command(
        {"argv": argv, "timeout_seconds": check.timeout_seconds},
        workspace,
        memory_limit=memory_limit,
    )
    pretty = " ".join(argv)
    if result.get("ok"):
        return f"{pretty} (ok)"
    output = str(result.get("output") or "")
    if len(output) > OUTPUT_CHARS:
        output = output[:OUTPUT_CHARS] + "\n… [check output truncated]"
    payload: dict = {
        "command": pretty,
        "exit_code": result.get("exit_code"),
        "output": output,
    }
    if result.get("limit"):
        payload["limit"] = result["limit"]
    if result.get("output_truncated"):
        payload["output_truncated"] = True
    return payload


def merge_into_edit_result(result: str, check: str | dict | None) -> str:
    """Attach a check outcome to an edit tool result without rewriting history."""
    if check is None:
        return result
    try:
        data = json.loads(result)
    except ValueError:
        return result
    if not isinstance(data, dict) or data.get("kind") != "edit" or not data.get("ok"):
        return result
    data["check"] = check
    return json.dumps(data, ensure_ascii=False)
