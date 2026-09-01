from hearthia.registry import Registry
from hearthia.rehearsal import rehearse
from hearthia.settings import Settings


class StubGateway:
    def __init__(self, running: list[dict] | None = None, chat_response: dict | None = None):
        self.running_list = running or []
        self.warmed: list[str] = []
        self.cooled: list[str] = []
        self._chat_response = chat_response or {"choices": [{"message": {"content": "OK"}}]}

    async def running(self):
        return self.running_list

    async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
        self.warmed.append(model_id)
        self.running_list = [*self.running_list, {"model": model_id, "state": "ready"}]
        return True

    async def cool(self, model_id: str | None) -> bool:
        self.cooled.append(model_id or "__all__")
        self.running_list = [m for m in self.running_list if m.get("model") != model_id]
        return True

    async def chat(self, body: dict, timeout: float = 600.0) -> dict:
        return self._chat_response

    async def close(self):
        pass


def _settings(tmp_path) -> Settings:
    s = Settings()
    s.paths.stack_dir = tmp_path
    (tmp_path / "llama-swap.yaml").write_text(
        """\
macros:
  llama-server: /opt/homebrew/bin/llama-server
  models_dir: /tmp/models
models:
  "big-coder":
    name: "Big Coder"
    cmd: |
      ${llama-server}
      --port ${PORT}
      --model ${models_dir}/big.gguf
      --ctx-size 32768
    ttl: 600
  "tiny-embed":
    name: "Tiny Embed"
    cmd: |
      ${llama-server}
      --port ${PORT}
      --model ${models_dir}/embed.gguf
      --embeddings
      --ctx-size 8192
"""
    )
    return s


def _reg(tmp_path) -> Registry:
    return Registry(tmp_path / "llama-swap.yaml", tmp_path / "backups")


async def test_rehearse_warms_canary_checks_and_cools_a_cold_model(tmp_path):
    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    gw = StubGateway()
    results = await rehearse(s, gw, reg.models())
    assert {r["model_id"]: r["status"] for r in results} == {"big-coder": "ok", "tiny-embed": "ok"}
    assert set(gw.warmed) == {"big-coder", "tiny-embed"}
    assert set(gw.cooled) == {"big-coder", "tiny-embed"}


async def test_rehearse_leaves_already_warm_model_running(tmp_path):
    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    gw = StubGateway(running=[{"model": "big-coder", "state": "ready"}])
    results = await rehearse(s, gw, reg.models())
    assert next(r for r in results if r["model_id"] == "big-coder")["status"] == "ok"
    assert "big-coder" not in gw.warmed
    assert "big-coder" not in gw.cooled  # never disturbed


async def test_rehearse_reports_canary_failure_and_still_cools(tmp_path):
    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    gw = StubGateway(chat_response={"choices": [{"message": {"content": ""}}]})
    results = await rehearse(s, gw, reg.models())
    assert all(r["status"] == "canary_failed" for r in results)
    assert set(gw.cooled) == {"big-coder", "tiny-embed"}  # cooled even on failure


async def test_rehearse_reports_blocked_when_budget_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    gw = StubGateway()
    results = await rehearse(s, gw, reg.models())
    assert all(r["status"] == "blocked" for r in results)
    assert gw.warmed == []


async def test_rehearse_reports_warm_failure(tmp_path):
    class FailingWarmGateway(StubGateway):
        async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
            return False

    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    gw = FailingWarmGateway()
    results = await rehearse(s, gw, reg.models())
    assert all(r["status"] == "warm_failed" for r in results)


async def test_rehearse_subset_still_accounts_for_other_running_models(tmp_path, monkeypatch):
    # tiny-embed is running but not in the rehearsed subset: the budget gate
    # must still see it as resident when deciding whether big-coder fits
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 2**30)
    s = _settings(tmp_path)
    reg = _reg(tmp_path)
    all_models = reg.models()
    targets = [m for m in all_models if m.id == "big-coder"]
    gw = StubGateway(running=[{"model": "tiny-embed", "rss": 2**30, "state": "ready"}])
    results = await rehearse(s, gw, targets, all_models=all_models)
    assert len(results) == 1
    assert results[0]["model_id"] == "big-coder"
    assert results[0]["status"] == "blocked"
