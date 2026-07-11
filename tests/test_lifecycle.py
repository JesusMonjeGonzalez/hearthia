from unittest.mock import patch

import httpx
import respx

from hearthia.gateway import Gateway
from hearthia.lifecycle import LifecycleEngine, parse_rule
from hearthia.registry import Registry
from hearthia.telemetry import Telemetry

BASE = "http://test-gw:9292"


def test_parse_rule_app():
    kind, target = parse_rule("app:Visual Studio Code")
    assert kind == "app"
    assert target == "Visual Studio Code"


def test_parse_rule_role():
    kind, target = parse_rule("role:chat")
    assert kind == "role"
    assert target == "chat"


def test_parse_rule_strips_whitespace():
    kind, target = parse_rule("  app:  Cursor  ")
    assert kind == "app"
    assert target == "Cursor"


@respx.mock
async def test_tick_warms_autocomplete_when_app_running(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    warm_route = respx.get(f"{BASE}/upstream/tiny-fim/health").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    rules = {"tiny-fim": "app:TestApp"}
    engine = LifecycleEngine(gw, reg, tel, rules)

    with patch("hearthia.lifecycle.app_alive", return_value=True):
        await engine.tick()

    assert warm_route.called
    await gw.close()


@respx.mock
async def test_tick_cools_autocomplete_when_app_stopped(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200,
        json={
            "running": [
                {"model": "big-coder", "state": "ready"},
                {"model": "tiny-fim", "state": "ready"},
            ]
        },
    )
    cool_route = respx.post(f"{BASE}/api/models/unload/tiny-fim").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    rules = {"tiny-fim": "app:TestApp"}
    engine = LifecycleEngine(gw, reg, tel, rules)

    with patch("hearthia.lifecycle.app_alive", return_value=False):
        await engine.tick()

    assert cool_route.called
    await gw.close()


@respx.mock
async def test_tick_warms_embed_when_chat_loaded(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    warm_route = respx.get(f"{BASE}/upstream/tiny-embed/health").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    rules = {"tiny-embed": "role:chat"}
    engine = LifecycleEngine(gw, reg, tel, rules)

    await engine.tick()

    assert warm_route.called
    await gw.close()


@respx.mock
async def test_tick_cools_embed_when_last_chat_unloaded(config_path, backups_dir):
    import time

    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "tiny-embed", "state": "ready"}]}
    )
    cool_route = respx.post(f"{BASE}/api/models/unload/tiny-embed").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    rules = {"tiny-embed": "role:chat"}
    engine = LifecycleEngine(gw, reg, tel, rules)

    engine._prev_role_alive = {"chat": True}
    engine._role_died_at = {"tiny-embed": time.time() - 400}  # beyond 300s grace
    await engine.tick()

    assert cool_route.called
    await gw.close()


@respx.mock
async def test_tick_does_not_cool_embed_if_direct_user_grace(config_path, backups_dir):
    """Embed stays warm during grace period after last chat model unloaded."""
    import time

    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "tiny-embed", "state": "ready"}]}
    )
    cool_route = respx.post(f"{BASE}/api/models/unload/tiny-embed").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    rules = {"tiny-embed": "role:chat"}
    engine = LifecycleEngine(gw, reg, tel, rules)

    engine._prev_role_alive = {"chat": True}
    engine._role_died_at = {"tiny-embed": time.time()}  # just died, within grace period
    await engine.tick()

    assert not cool_route.called
    await gw.close()


@respx.mock
async def test_ttl_poisoning_regression(config_path, backups_dir):
    """tick() must check running state via /running, never via /upstream polling."""
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )
    upstream_route = respx.get(f"{BASE}/upstream/big-coder/health").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, {})

    await engine.tick()

    assert not upstream_route.called
    await gw.close()


@respx.mock
async def test_tick_no_rules_does_nothing(config_path, backups_dir):
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "big-coder", "state": "ready"}]}
    )

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, {})

    await engine.tick()
    await gw.close()


@respx.mock
async def test_tick_gateway_down_does_not_raise(config_path, backups_dir):
    respx.get(f"{BASE}/running").mock(side_effect=httpx.ConnectError("down"))

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, {"tiny-fim": "app:TestApp"})

    with patch("hearthia.lifecycle.app_alive", return_value=True):
        await engine.tick()

    await gw.close()


def test_notify_crash_loop_triggers_notification():
    import time

    gw = Gateway(BASE)
    reg = Registry.__new__(Registry)
    tel = Telemetry(gw)
    now = time.time()
    tel._crashes = [now - 10, now - 5, now - 1]
    engine = LifecycleEngine(gw, reg, tel, {})

    with patch("subprocess.Popen") as mock_popen:
        engine.notify_crash_loop()

    assert mock_popen.called


def test_notify_crash_loop_does_not_notify_below_threshold():
    import time

    gw = Gateway(BASE)
    reg = Registry.__new__(Registry)
    tel = Telemetry(gw)
    now = time.time()
    tel._crashes = [now - 10, now - 5]
    engine = LifecycleEngine(gw, reg, tel, {})

    with patch("subprocess.Popen") as mock_popen:
        engine.notify_crash_loop()

    assert not mock_popen.called


NO_ROLES_YAML = """\
models:
  "chatty":
    name: "Chatty"
    cmd: |
      llama-server --port ${PORT} --model /tmp/chatty.gguf
    ttl: 600
  "embed":
    name: "Embed"
    cmd: |
      llama-server --port ${PORT} --model /tmp/embed.gguf --embeddings
  "fim":
    name: "FIM"
    cmd: |
      llama-server --port ${PORT} --model /tmp/fim.gguf
"""


@respx.mock
async def test_role_chat_falls_back_when_no_roles_declared(tmp_path, backups_dir):
    """Adopted configs rarely declare metadata.roles; role:chat must still work.

    Fallback: chat = every non-embedding model that isn't itself lifecycle-managed.
    """
    cfg = tmp_path / "llama-swap.yaml"
    cfg.write_text(NO_ROLES_YAML)
    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "chatty", "state": "ready"}]}
    )
    warm_route = respx.get(f"{BASE}/upstream/embed/health").respond(200)

    gw = Gateway(BASE)
    reg = Registry(cfg, backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, {"embed": "role:chat", "fim": "app:TestApp"})

    with patch("hearthia.lifecycle.app_alive", return_value=False):
        await engine.tick()

    assert warm_route.called
    await gw.close()


@respx.mock
async def test_role_grace_timers_are_independent_per_follower(config_path, backups_dir):
    """Two role followers must not share one grace timer."""
    import time

    respx.get(f"{BASE}/running").respond(
        200, json={"running": [{"model": "tiny-embed", "state": "ready"}]}
    )
    cool_route = respx.post(f"{BASE}/api/models/unload/tiny-embed").respond(200)

    gw = Gateway(BASE)
    reg = Registry(config_path, backups_dir)
    tel = Telemetry(gw)
    engine = LifecycleEngine(gw, reg, tel, {"tiny-embed": "role:chat"})

    engine._prev_role_alive = {"chat": True}
    engine._role_died_at["tiny-embed"] = time.time() - 400
    await engine.tick()

    assert cool_route.called
    assert isinstance(engine._role_died_at, dict)
    await gw.close()
