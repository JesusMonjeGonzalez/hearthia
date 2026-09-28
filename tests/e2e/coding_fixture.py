"""Scripted model decisions; real tools, files and unittest commands for browser tests."""

import json
import sys
from pathlib import Path

from hearthia.demo import create_demo_app


def create_app(directory: Path, port: int):
    app = create_demo_app(directory, port)
    project = directory / "coding-project"
    project.mkdir()
    (project / "calc.py").write_text("# user comment\ndef add(a, b):\n    return a - b\n")
    (project / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\nclass Addition(unittest.TestCase):\n"
        "    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
    )
    (project / "AGENTS.md").write_text("Preserve the user comment; verify with unittest.")
    original = app.state.gateway.chat_stream
    command = {"argv": [sys.executable, "-B", "-m", "unittest", "test_calc", "-v"]}
    plan = [
        (
            "update_plan",
            {"steps": ["read calc.py", "run the failing test", "fix add()", "re-run tests"]},
        ),
        ("read_file", {"path": "calc.py"}),
        ("run_command", command),
        ("edit_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
        ("read_file", {"path": "calc.py"}),
        ("run_command", command),
        (
            "update_plan",
            {
                "steps": ["read calc.py", "run the failing test", "fix add()", "re-run tests"],
                "done": [1, 2, 3, 4],
            },
        ),
    ]

    async def scripted(body):
        messages = json.loads(body)["messages"]
        latest = max(i for i, message in enumerate(messages) if message["role"] == "user")
        # On the wire, context arrives as a second trailing user message and
        # its internal tag is stripped. Take the most recent user turn that
        # carries a fixture marker, not the first marker in the whole history.
        markers = ("HEARTHIA_FIXTURE_TASK", "HEARTHIA_LONG_COMMAND_TASK")
        content = ""
        for message in reversed(messages):
            if message["role"] == "user" and any(m in message["content"] for m in markers):
                content = message["content"]
                break
        task = next((marker for marker in markers if marker in content), "")
        if not task:
            async for chunk in original(body):
                yield chunk
            return
        if task == "HEARTHIA_LONG_COMMAND_TASK":
            active_plan = [
                (
                    "run_command",
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            "import os,time; from pathlib import Path; "
                            "Path('running.pid').write_text(str(os.getpid())); time.sleep(60)",
                        ]
                    },
                )
            ]
        else:
            active_plan = plan
        step = sum(message["role"] == "tool" for message in messages[latest:])
        if step < len(active_plan):
            name, arguments = active_plan[step]
            choice = {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"fixture-{step}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        else:
            choice = {
                "delta": {"content": "Fixture fixed; unittest passed."},
                "finish_reason": "stop",
            }
        yield ("data: " + json.dumps({"choices": [choice]}) + "\n\ndata: [DONE]\n\n").encode()

    app.state.gateway.chat_stream = scripted
    return app
