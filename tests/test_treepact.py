import json
import subprocess
from pathlib import Path

import pytest

from hearthia.settings import TreePactSettings
from hearthia.treepact import (
    TreePactBridge,
    TreePactBridgeError,
    TreePactReviewNotFoundError,
)


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "treepact"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o700)
    return executable


def test_resolves_configured_executable(tmp_path):
    executable = _executable(tmp_path)
    bridge = TreePactBridge.from_settings(TreePactSettings(executable=executable))
    assert bridge.executable == executable.resolve()


def test_missing_executable_fails_closed(monkeypatch):
    monkeypatch.setattr("hearthia.treepact.shutil.which", lambda _: None)
    with pytest.raises(TreePactBridgeError, match="not installed"):
        TreePactBridge.from_settings(TreePactSettings())


def test_rejects_unknown_version(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    monkeypatch.setattr(
        "hearthia.treepact.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "treepact 0.2.0\n", ""),
    )
    with pytest.raises(TreePactBridgeError, match="unsupported"):
        bridge.verify_version()


def test_run_uses_fixed_argv_and_propagates_exit_code(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-1] == "--version":
            return subprocess.CompletedProcess(argv, 0, "treepact 0.1.0 | pact schema 1\n", "")
        return subprocess.CompletedProcess(argv, 16)

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    task = 'fix tests; rm -rf "$HOME"'
    result = bridge.run(tmp_path, task, mode="repair", max_attempts=2, max_minutes=10)

    assert result == 16
    argv, kwargs = calls[1]
    assert task in argv
    assert argv.count(task) == 1
    assert argv[:3] == [str(bridge.executable), "run", task]
    assert ["--runtime", "native"] == argv[argv.index("--runtime") : argv.index("--runtime") + 2]
    assert kwargs == {"check": False}


def test_validate_canonicalizes_repo_path(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    monkeypatch.setattr(TreePactBridge, "invoke", lambda self, args: 0)
    assert bridge.validate(tmp_path / ".." / tmp_path.name) == 0


def test_review_commands_use_bounded_argv(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    calls = []
    monkeypatch.setattr(TreePactBridge, "invoke", lambda self, args: calls.append(args) or 0)

    assert bridge.status("run_123") == 0
    assert bridge.diff("run_123", stat=True) == 0
    assert bridge.evidence("run_123", output_format="json", verify_hashes=True) == 0
    assert bridge.verify("run_123") == 0

    assert calls == [
        ["status", "run_123"],
        ["diff", "run_123", "--stat"],
        ["evidence", "run_123", "--format", "json", "--verify-hashes"],
        ["verify", "run_123", "--check-artifacts", "--check-events"],
    ]


def test_diff_rejects_conflicting_views(tmp_path):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    with pytest.raises(TreePactBridgeError, match="cannot be used together"):
        bridge.diff("run_123", stat=True, name_only=True)


def test_evidence_rejects_unknown_format(tmp_path):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    with pytest.raises(TreePactBridgeError, match="format"):
        bridge.evidence("run_123", output_format="yaml")


@pytest.mark.parametrize("mode", ["write", "autonomous"])
def test_run_rejects_unknown_modes(tmp_path, mode):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    with pytest.raises(TreePactBridgeError, match="observe or repair"):
        bridge.run(tmp_path, "task", mode=mode)


RUN_ID = "run_" + "a" * 32
_LIST_DOC = {
    "schema": "treepact.review",
    "schema_version": 1,
    "kind": "run_list",
    "generated_at": "2026-08-30T00:00:00Z",
    "runs": [],
}
_DETAIL_DOC = {
    "schema": "treepact.review",
    "schema_version": 1,
    "kind": "run_detail",
    "generated_at": "2026-08-30T00:00:00Z",
    "run": {"run_id": RUN_ID},
}


def _version_result(argv):
    return subprocess.CompletedProcess(argv, 0, "treepact 0.1.0 | pact schema 1\n", "")


def test_review_runs_uses_bounded_argv_and_allowlisted_env(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leak-me-not")
    calls = []

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(_LIST_DOC), "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    document = bridge.review_runs(20)

    assert document == _LIST_DOC
    (argv, kwargs) = calls[0]
    assert argv == [str(bridge.executable), "review", "--schema-version", "1", "--limit", "20"]
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] == 5.0
    assert "SSH_AUTH_SOCK" not in kwargs["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
    assert kwargs["env"]["HOME"] == str(tmp_path)


def test_review_run_uses_bounded_argv(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    calls = []

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(_DETAIL_DOC), "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    document = bridge.review_run(RUN_ID)

    assert document == _DETAIL_DOC
    assert calls[0] == [
        str(bridge.executable),
        "review",
        "--schema-version",
        "1",
        "--run-id",
        RUN_ID,
    ]


@pytest.mark.parametrize("limit", [0, 101])
def test_review_runs_rejects_out_of_range_limit(tmp_path, limit):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    with pytest.raises(TreePactBridgeError, match="between 1 and 100"):
        bridge.review_runs(limit)


def test_review_run_rejects_malformed_run_id(tmp_path):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    with pytest.raises(TreePactBridgeError, match="run_<32 lowercase hex>"):
        bridge.review_run("run_BAD")


def test_review_run_not_found_raises_dedicated_error(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(
            argv, 20, "", "error: [run_not_found] TreePact run was not found\n"
        )

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactReviewNotFoundError):
        bridge.review_run(RUN_ID)


def test_review_generic_failure_does_not_leak_stderr(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(
            argv, 20, "", "error: [storage_incompatible] /Users/secret/path leaked\n"
        )

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError) as excinfo:
        bridge.review_runs(20)
    assert "/Users/secret/path" not in str(excinfo.value)


def test_review_times_out(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        raise subprocess.TimeoutExpired(argv, 5.0)

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError, match="timed out"):
        bridge.review_runs(20)


def test_review_rejects_oversized_output(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    huge = json.dumps({**_LIST_DOC, "runs": [{"padding": "x" * (2 * 1024 * 1024)}]})

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(argv, 0, huge, "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError, match="size limit"):
        bridge.review_runs(20)


def test_review_rejects_invalid_json(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(argv, 0, "not json", "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError, match="invalid JSON"):
        bridge.review_runs(20)


def test_review_rejects_incompatible_schema_version(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")
    bad = {**_LIST_DOC, "schema_version": 2}

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(bad), "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError, match="incompatible contract version"):
        bridge.review_runs(20)


def test_review_runs_rejects_mismatched_document_kind(tmp_path, monkeypatch):
    bridge = TreePactBridge(_executable(tmp_path), "0.1.0")

    def fake_run(argv, **kwargs):
        if argv[-1] == "--version":
            return _version_result(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(_DETAIL_DOC), "")

    monkeypatch.setattr("hearthia.treepact.subprocess.run", fake_run)
    with pytest.raises(TreePactBridgeError, match="unexpected document kind"):
        bridge.review_runs(20)
