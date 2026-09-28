import asyncio
import hashlib
import json
import os
import stat
import sys

import psutil
import pytest

from hearthia.coding_tools import apply_edit
from hearthia.commands import OUTPUT_BYTES, OutputBuffer, run_command
from hearthia.tool_runtime import run_tool


def test_exact_edit_preserves_unrelated_content_and_mode(tmp_path):
    target = tmp_path / "code.py"
    target.write_bytes(b"# user's change\r\nx = 1\r\n")
    target.chmod(0o755)
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    result = apply_edit(
        "edit_file",
        {"path": "code.py", "old_text": "x = 1", "new_text": "x = 2", "expected_sha256": before},
        str(tmp_path),
    )
    assert target.read_bytes() == b"# user's change\r\nx = 2\r\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert result["before_sha256"] == before
    assert "-x = 1" in result["diff"] and "+x = 2" in result["diff"]
    assert not list(tmp_path.glob(".hearthia-edit-*"))


@pytest.mark.parametrize(
    "args",
    [
        {"old_text": "absent", "new_text": "new"},
        {"old_text": "repeat", "new_text": "new"},
        {"old_text": "repeat repeat", "new_text": "new", "expected_sha256": "wrong"},
        {"old_text": "", "new_text": "new"},
    ],
)
def test_stale_or_ambiguous_edit_is_rejected(tmp_path, args):
    target = tmp_path / "code.py"
    target.write_text("repeat repeat")
    with pytest.raises(ValueError):
        apply_edit("edit_file", {"path": "code.py", **args}, str(tmp_path))
    assert target.read_text() == "repeat repeat"


def test_create_never_overwrites_and_publication_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "new.py"
    result = apply_edit("create_file", {"path": "new.py", "content": "new content"}, str(tmp_path))
    assert result["ok"] and target.read_text() == "new content"
    with pytest.raises(ValueError, match="already exists"):
        apply_edit("create_file", {"path": "new.py", "content": "overwrite"}, str(tmp_path))

    real_link = os.link

    def raced_link(source, destination):
        destination.write_text("concurrent user work")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        apply_edit("create_file", {"path": "race.py", "content": "overwrite"}, str(tmp_path))
    assert (tmp_path / "race.py").read_text() == "concurrent user work"
    assert not list(tmp_path.glob(".hearthia-edit-*"))


def test_failed_atomic_replace_leaves_original_intact(tmp_path, monkeypatch):
    target = tmp_path / "code.py"
    target.write_text("old")

    def fail(*args):
        raise OSError("test publication failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        apply_edit(
            "edit_file", {"path": "code.py", "old_text": "old", "new_text": "new"}, str(tmp_path)
        )
    assert target.read_text() == "old"
    assert not list(tmp_path.glob(".hearthia-edit-*"))


def test_file_tools_reject_escape_symlinks_binary_and_repository_metadata(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    (workspace / "link").symlink_to(tmp_path / "outside.txt")
    (workspace / ".git").mkdir()
    (workspace / "binary").write_bytes(b"binary\0data")
    for path in ("../outside.txt", "link", ".git/config", "binary"):
        with pytest.raises(ValueError):
            apply_edit(
                "edit_file",
                {"path": path, "old_text": "outside", "new_text": "new"},
                str(workspace),
            )
    assert (tmp_path / "outside.txt").read_text() == "outside"


@pytest.mark.parametrize(
    "name,args",
    [
        ("create_file", {"path": "forbidden", "content": "no"}),
        ("edit_file", {"path": "forbidden", "old_text": "no", "new_text": "yes"}),
        ("run_command", {"argv": [sys.executable, "-c", "open('forbidden','w').write('no')"]}),
    ],
)
async def test_read_mode_denies_mutation_even_if_model_requests_it(tmp_path, name, args):
    result = await run_tool(
        {"function": {"name": name, "arguments": json.dumps(args)}}, workspace=str(tmp_path)
    )
    assert result.startswith("Error:")
    assert not (tmp_path / "forbidden").exists()


async def test_worker_reports_failed_edits_as_errors(tmp_path):
    (tmp_path / "a").write_text("existing")
    result = await run_tool(
        {"function": {"name": "create_file", "arguments": '{"path":"a","content":"new"}'}},
        workspace=str(tmp_path),
        mode="build",
    )
    assert result.startswith("Error:")
    assert (tmp_path / "a").read_text() == "existing"


def test_output_buffer_is_bounded_and_keeps_head_and_tail():
    buffer = OutputBuffer(32)
    for data in (b"BEGIN", b"x" * 1000, b"END"):
        buffer.feed(data)
    assert len(buffer.head) + len(buffer.tail) <= 32
    assert buffer.text().startswith("BEGIN") and buffer.text().endswith("END")
    assert "truncated" in buffer.text()


async def test_command_exit_output_cwd_and_literal_arguments(tmp_path):
    result = await run_command(
        {
            "argv": [
                sys.executable,
                "-c",
                "import os,sys; print(os.getcwd()); print(sys.argv[1]); sys.exit(7)",
                "$(touch forbidden)",
            ],
            "timeout_seconds": 5,
        },
        str(tmp_path),
    )
    assert result["exit_code"] == 7 and not result["ok"]
    assert str(tmp_path) in result["output"]
    assert "$(touch forbidden)" in result["output"]
    assert not (tmp_path / "forbidden").exists()


async def test_command_drains_large_output_without_growing_history(tmp_path):
    result = await run_command(
        {"argv": [sys.executable, "-c", "print('FIRST'); print('x'*2000000); print('LAST')"]},
        str(tmp_path),
    )
    assert result["ok"] and result["output_truncated"]
    assert len(result["output"].encode()) < OUTPUT_BYTES + 100
    assert result["output"].startswith("FIRST") and result["output"].endswith("LAST\n")


async def test_command_wall_time_and_sampled_memory_limit(tmp_path):
    command = {"argv": [sys.executable, "-c", "import time; time.sleep(60)"], "timeout_seconds": 1}
    timed = await run_command(command, str(tmp_path))
    assert timed["limit"] == "timeout" and not timed["ok"]
    assert timed["duration_seconds"] < 5
    memory = await run_command(command, str(tmp_path), memory_limit=1)
    assert memory["limit"] == "memory_limit" and not memory["ok"]


@pytest.mark.parametrize("cancel", [True, False])
async def test_command_cleans_up_children_on_cancel_and_normal_exit(tmp_path, cancel):
    marker = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "open('child.pid','w').write(str(p.pid)); "
        + ("time.sleep(60)" if cancel else "time.sleep(0.1)")
    )
    task = asyncio.create_task(run_command({"argv": [sys.executable, "-c", script]}, str(tmp_path)))
    for _ in range(200):
        if marker.exists() and marker.read_text():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    child = int(marker.read_text())
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task)["ok"]
    for _ in range(100):
        try:
            if psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Command child survived cleanup")


@pytest.mark.parametrize(
    "name,content,expect",
    [
        ("ok.py", "x = 1\n", None),
        ("broken.py", "def f(:\n    pass\n", "line 1"),
        ("ok.json", '{"a": 1}', None),
        ("broken.json", '{"a": }', "Expecting value"),
        ("ok.toml", "a = 1\n", None),
        ("broken.toml", "a = = 1\n", "Invalid value"),
        ("notes.txt", "anything (:\n", None),  # not a checkable extension
    ],
)
def test_edit_reports_syntax_only_when_it_fails(tmp_path, name, content, expect):
    result = apply_edit("create_file", {"path": name, "content": content}, str(tmp_path))
    assert result["ok"] is True  # the edit is never blocked
    if expect is None:
        assert "syntax_error" not in result
    else:
        assert expect in result["syntax_error"]
        assert "no longer parses" in result["note"]
    assert (tmp_path / name).read_text() == content


def test_edit_that_breaks_an_existing_python_file_is_flagged(tmp_path):
    (tmp_path / "code.py").write_text("def f():\n    return 1\n")
    result = apply_edit(
        "edit_file",
        {"path": "code.py", "old_text": "return 1", "new_text": "return ("},
        str(tmp_path),
    )
    assert "syntax_error" in result
    assert (tmp_path / "code.py").read_text() == "def f():\n    return (\n"


def test_command_spec_prepends_project_venv_and_common_bins(tmp_path, monkeypatch):
    from hearthia.commands import command_spec

    workspace = tmp_path / "proj"
    (workspace / ".venv" / "bin").mkdir(parents=True)
    spec, env, cwd = command_spec({"argv": ["python", "-V"]}, str(workspace))
    entries = env["PATH"].split(":")
    assert entries[0] == str(workspace / ".venv" / "bin")
    assert "/opt/homebrew/bin" in entries and "/usr/local/bin" in entries
    assert cwd == workspace
    # The original PATH survives at the tail.
    import os

    assert env["PATH"].endswith(os.environ.get("PATH", ""))


async def test_missing_executable_reports_command_not_found_cleanly(tmp_path):
    from hearthia.commands import run_command

    result = await run_command(
        {"argv": ["definitely-not-a-real-binary-xyz"], "timeout_seconds": 10}, str(tmp_path)
    )
    assert result["exit_code"] == 127
    assert "not found" in result["output"]
    assert "Traceback" not in result["output"]
    assert "PATH searched" in result["output"]
