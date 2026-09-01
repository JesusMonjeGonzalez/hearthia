import subprocess

from hearthia.sleep_guard import SleepGuard


class _FakeProc:
    def __init__(self) -> None:
        self.pid = 12345
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_sync_starts_caffeinate_when_a_model_warms(monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    guard = SleepGuard()
    assert guard.active is False
    guard.sync(any_warm=True)
    assert guard.active is True
    assert calls == [["/usr/bin/caffeinate", "-s", "-i"]]


def test_sync_is_idempotent_while_still_warm(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: (calls.append(1), _FakeProc())[1])
    guard = SleepGuard()
    guard.sync(any_warm=True)
    guard.sync(any_warm=True)
    guard.sync(any_warm=True)
    assert len(calls) == 1  # only one caffeinate process for a continuously-warm stretch


def test_sync_stops_caffeinate_once_everything_cools(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    guard = SleepGuard()
    guard.sync(any_warm=True)
    assert guard.active is True
    guard.sync(any_warm=False)
    assert guard.active is False


def test_sync_never_started_is_a_no_op_when_nothing_warm():
    guard = SleepGuard()
    guard.sync(any_warm=False)
    assert guard.active is False


def test_stop_is_safe_to_call_when_never_started():
    guard = SleepGuard()
    guard.stop()  # must not raise
    assert guard.active is False


def test_start_failure_is_caught(monkeypatch):
    def boom(cmd, **kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(subprocess, "Popen", boom)
    guard = SleepGuard()
    guard.sync(any_warm=True)  # must not raise
    assert guard.active is False
