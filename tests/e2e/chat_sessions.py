"""Real-browser session checks on an isolated synthetic daemon (no model loads).

uvx --from playwright python tests/e2e/chat_sessions.py
Uses installed Chrome when present; otherwise Playwright's installed Chromium.
"""

import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="hearthia-chat-e2e-") as directory:
        command = (
            "import sys,uvicorn; from pathlib import Path; "
            "sys.path.insert(0,sys.argv[3]); from coding_fixture import create_app; "
            "uvicorn.run(create_app(Path(sys.argv[1]),int(sys.argv[2])), "
            "host='127.0.0.1',port=int(sys.argv[2]),log_level='error')"
        )
        process = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-c",
                command,
                directory,
                str(port),
                str(ROOT / "tests/e2e"),
            ],
            cwd=ROOT,
        )
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(base + "/api/conversations", timeout=1).close()
                    break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Demo daemon exited") from None
                    time.sleep(0.1)
            with sync_playwright() as p:
                channel = "chrome" if Path("/Applications/Google Chrome.app").exists() else None
                browser = p.chromium.launch(channel=channel)
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"{base}/?chat=1&workspace={directory}&mode=read")
                expect(page.locator("#tab-chat")).to_have_class(re.compile(r"active"))
                expect(page.locator("#chat-workspace")).to_have_value(directory)
                expect(page.locator("#chat-mode")).to_have_value("read")
                page.goto(base)
                page.click('[data-tab="chat"]')
                page.click("#conv-new")
                expect(page.locator(".conv-item")).to_have_count(1)
                page.fill("#chat-workspace", directory)
                page.fill("#chat-input", "Hello persistent conversation")
                page.click("#chat-send")
                expect(page.locator("#chat-stop")).to_be_visible()
                expect(page.locator("#chat-stop")).to_be_hidden(timeout=30_000)
                expect(page.locator("#chat-log")).to_contain_text("Hello from the Hearthia demo")
                page.reload()
                page.click('[data-tab="chat"]')
                expect(page.locator("#chat-log")).to_contain_text("Hello persistent conversation")
                expect(page.locator("#chat-log")).to_contain_text("Hello from the Hearthia demo")
                # Occupancy + projection come from the recorded turn usage.
                expect(page.locator("#chat-context")).to_contain_text("window")
                expect(page.locator("#chat-context")).to_contain_text("tok free")
                expect(page.locator("#chat-workspace")).to_have_value(directory)

                legacy = [
                    {
                        "title": "Legacy fixture",
                        "messages": [
                            {
                                "role": "user" if i % 2 == 0 else "assistant",
                                "content": f"Message {i}",
                            }
                            for i in range(90)
                        ],
                    }
                ]
                page.evaluate(
                    "data => localStorage.setItem('hearthia.convs', JSON.stringify(data))", legacy
                )
                page.click("#conv-import")
                expect(page.locator(".conv-item")).to_have_count(2)
                page.click("#conv-import")
                expect(page.locator("#chat-stats")).to_contain_text("Imported 1 conversations")
                expect(page.locator(".conv-item")).to_have_count(2)
                assert json.loads(page.evaluate("localStorage.getItem('hearthia.convs')")) == legacy
                page.get_by_text("Legacy fixture", exact=True).click()
                expect(page.locator("#chat-log .msg")).to_have_count(40)
                expect(page.locator("#chat-log")).to_contain_text("Message 89")
                page.click("#chat-older")
                expect(page.locator("#chat-log")).to_contain_text("Message 10")
                expect(page.locator("#chat-log .msg")).to_have_count(40)
                assert page.eval_on_selector("#chat-log", "el => el.scrollTop") == 0
                page.click("#chat-latest")
                expect(page.locator("#chat-log")).to_contain_text("Message 89")
                with page.expect_download() as download:
                    page.click("#conv-export")
                exported = Path(download.value.path()).read_text()
                assert "Message 0" in exported and "Message 89" in exported

                page.fill("#chat-input", "Stop this turn")
                page.click("#chat-send")
                expect(page.locator("#chat-stop")).to_be_visible()
                page.click("#chat-stop")
                expect(page.locator("#chat-send")).to_be_visible()
                page.click("#conv-new")
                expect(page.locator(".conv-item")).to_have_count(3)
                project = Path(directory) / "coding-project"
                page.fill("#chat-workspace", str(project))
                page.select_option("#chat-mode", "build")
                page.select_option("#chat-model", "qwen3.8-27b")
                page.fill("#chat-input", "HEARTHIA_FIXTURE_TASK")
                page.click("#chat-send")
                expect(page.locator("#chat-log")).to_contain_text("Fixture fixed", timeout=30_000)
                expect(page.locator("#chat-stop")).to_be_hidden()
                expect(page.locator(".tool-edit")).to_have_count(1)
                expect(page.locator(".tool-command")).to_have_count(2)
                expect(page.locator(".tool-command summary").first).to_contain_text("Failed")
                expect(page.locator(".tool-command summary").last).to_contain_text("Passed")
                expect(page.locator(".tool-edit")).to_contain_text("+    return a + b")
                expect(page.locator("#chat-plan")).to_be_visible()
                expect(page.locator("#chat-plan-summary")).to_contain_text("4/4 done")
                expect(page.locator("#chat-plan-steps")).to_contain_text("✔ fix add()")
                expect(page.locator(".tool-plan").last).to_contain_text("Plan · 4/4 done")
                expect(page.locator("#chat-autocontinue")).to_be_visible()
                assert (
                    project / "calc.py"
                ).read_text() == "# user comment\ndef add(a, b):\n    return a + b\n"
                page.reload()
                page.click('[data-tab="chat"]')
                expect(page.locator("#chat-mode")).to_have_value("build")
                expect(page.locator(".tool-edit")).to_have_count(1)
                page.fill("#chat-input", "HEARTHIA_LONG_COMMAND_TASK")
                page.click("#chat-send")
                marker = project / "running.pid"
                for _ in range(200):
                    if marker.exists() and marker.read_text():
                        break
                    time.sleep(0.025)
                assert marker.exists(), "Foreground command did not start"
                command_pid = int(marker.read_text())
                page.click("#chat-stop")
                expect(page.locator("#chat-send")).to_be_visible()
                for _ in range(100):
                    try:
                        os.kill(command_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.025)
                else:
                    raise AssertionError("Foreground command survived browser Stop")
                # Full-text search across conversations, then jump to the hit.
                page.fill("#conv-search", "Message 42")
                expect(page.locator("#conv-items .conv-item").first).to_contain_text(
                    "Legacy fixture"
                )
                expect(page.locator("#conv-items .conv-snippet").first).to_contain_text(
                    "Message 42"
                )
                page.locator("#conv-items .conv-item").first.click()
                expect(page.locator("#chat-log")).to_contain_text("Message 42")
                page.fill("#conv-search", "")

                # Fork and retry: branch a conversation, and re-run its last
                # question in a fresh copy without touching the original.
                page.get_by_text("Legacy fixture", exact=True).click()
                page.click("#conv-fork")
                expect(page.locator("#chat-stats")).to_contain_text("Forked")
                expect(page.locator(".conv-item")).to_have_count(4)
                page.click("#conv-retry")
                expect(page.locator("#chat-log")).to_contain_text("Message 88")
                expect(page.locator("#chat-log")).to_contain_text(
                    "Hello from the Hearthia demo", timeout=30_000
                )
                expect(page.locator(".conv-item")).to_have_count(5)
                # Keyboard shortcut: Cmd/Ctrl+K starts a new conversation.
                expect(page.locator("#chat-send")).to_be_visible(timeout=30_000)
                before = page.locator(".conv-item").count()
                page.keyboard.press("Meta+k")
                expect(page.locator(".conv-item")).to_have_count(before + 1)

                # `hearth chat --new` deep link: a clean conversation on boot
                # (checked last so the earlier count assertions stay stable).
                count_before_deeplink = page.locator(".conv-item").count()
                page.goto(f"{base}/?chat=1&new=1&workspace={directory}&mode=build")
                expect(page.locator("#chat-log")).to_contain_text("Choose a project")
                expect(page.locator("#chat-workspace")).to_have_value(directory)
                expect(page.locator(".conv-item")).to_have_count(count_before_deeplink + 1)
                assert not errors, errors
                browser.close()
            print(
                "PASS: durable chat, reload, workspace, idempotent migration, "
                "pagination, search, export, stop, fork, retry, verification, "
                "real edit/test/fix workflow, command cancellation"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
