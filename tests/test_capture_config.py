"""Configurable brain filing: folders and prompt from [brain] settings."""

from pathlib import Path

from hearthia.brain.capture import build_schema, load_prompt, write_note
from hearthia.settings import Settings


def test_build_schema_uses_custom_folders():
    schema = build_schema(["00 Inbox", "99 Somewhere"])
    enum = schema["properties"]["folder"]["enum"]
    assert enum == ["00 Inbox", "99 Somewhere"]


def test_load_prompt_default_without_path():
    assert "{text}" in load_prompt(None)


def test_load_prompt_from_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("File this: {text}")
    assert load_prompt(p) == "File this: {text}"


def test_load_prompt_falls_back_without_placeholder(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("no placeholder here")
    assert "{text}" in load_prompt(p)


def test_load_prompt_falls_back_on_missing_file(tmp_path):
    assert "{text}" in load_prompt(tmp_path / "missing.txt")


def test_write_note_uses_custom_folders(tmp_path):
    folders = ["10 Landing", "20 Deep"]
    path = write_note(tmp_path, "hello", None, folders=folders)
    assert path.startswith(str(tmp_path / "10 Landing"))


def test_write_note_rejects_folder_outside_config(tmp_path):
    folders = ["10 Landing", "20 Deep"]
    path = write_note(
        tmp_path,
        "hello",
        {"title": "T", "tags": ["x"], "folder": "00 Inbox"},
        folders=folders,
    )
    assert path.startswith(str(tmp_path / "10 Landing"))


def test_settings_parse_loadouts_and_brain(tmp_path, monkeypatch):
    cfg = tmp_path / "hearthia-config.toml"  # conftest points HEARTHIA_CONFIG here
    cfg.write_text(
        """\
[brain]
vault = "/tmp/vault"
folders = ["Inbox", "Projects"]
prompt_path = "/tmp/prompt.txt"

[loadouts.coding]
description = "coder + embeds"
models = ["qwen-coder-30b", "qwen3-embedding-0.6b"]
"""
    )
    s = Settings()
    assert s.brain.folders == ["Inbox", "Projects"]
    assert s.brain.prompt_path == Path("/tmp/prompt.txt")
    assert s.loadouts["coding"].models == [
        "qwen-coder-30b",
        "qwen3-embedding-0.6b",
    ]
    assert s.loadouts["coding"].description == "coder + embeds"
