from hearthia.loadouts import (
    defined_loadouts,
    get_loadout,
    loadout_cool,
    loadout_load,
    loadout_plan,
)
from hearthia.registry import Registry
from hearthia.settings import LoadoutSettings, Settings

GIB = 2**30


class StubGateway:
    def __init__(self, running: list[dict] | None = None):
        self.running_list = running or []
        self.warmed: list[str] = []
        self.cooled: list[str] = []

    async def running(self):
        return self.running_list

    async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
        self.warmed.append(model_id)
        return True

    async def cool(self, model_id: str | None) -> bool:
        self.cooled.append(model_id or "__all__")
        return True

    async def close(self):
        pass


def _settings(tmp_path, loadouts: dict) -> Settings:
    s = Settings()
    s.paths.stack_dir = tmp_path
    s.loadouts = loadouts
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


def _loadouts(models: list[str]) -> dict:
    return {"coding": LoadoutSettings(models=models, description="test set")}


def _reg(tmp_path) -> Registry:
    return Registry(tmp_path / "llama-swap.yaml", tmp_path / "backups")


def test_defined_loadouts_filters_empty():
    s = Settings()
    s.loadouts = {
        "coding": LoadoutSettings(models=["a"], description=""),
        "empty": LoadoutSettings(models=[], description="nothing"),
    }
    assert set(defined_loadouts(s)) == {"coding"}
    assert get_loadout(s, "empty") is None


async def test_loadout_load_warms_each_member(tmp_path):
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    gw = StubGateway()
    result = await loadout_load(s, gw, _reg(tmp_path), "coding")
    assert result["ok"] is True
    assert result["warmed"] == ["big-coder", "tiny-embed"]
    assert result["skipped"] == []
    assert gw.warmed == ["big-coder", "tiny-embed"]


async def test_loadout_load_skips_already_warm(tmp_path):
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    gw = StubGateway(running=[{"model": "big-coder", "state": "ready"}])
    result = await loadout_load(s, gw, _reg(tmp_path), "coding")
    assert result["ok"] is True
    assert result["skipped"] == ["big-coder"]
    assert gw.warmed == ["tiny-embed"]


async def test_loadout_load_refuses_set_that_does_not_fit(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.loadouts.wired_limit_bytes", lambda total: GIB)
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: GIB)
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    gw = StubGateway()
    result = await loadout_load(s, gw, _reg(tmp_path), "coding")
    assert result["ok"] is False
    assert "does not fit" in result["error"]
    assert isinstance(result["advice"]["options"], list)
    assert gw.warmed == []


async def test_loadout_load_unknown_name(tmp_path):
    s = _settings(tmp_path, {})
    result = await loadout_load(s, StubGateway(), _reg(tmp_path), "nope")
    assert result["ok"] is False
    assert "not defined" in result["error"]


async def test_loadout_plan_reports_fit(tmp_path):
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    result = await loadout_plan(s, StubGateway(), _reg(tmp_path), "coding")
    assert "error" not in result
    assert result["fits"] is True
    assert result["total_bytes"] == 2 * 512 * 1024**2  # two file-size floor guesses


async def test_loadout_cool_only_warm_members(tmp_path):
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    gw = StubGateway(running=[{"model": "tiny-embed", "state": "ready"}])
    result = await loadout_cool(s, gw, _reg(tmp_path), "coding")
    assert result["ok"] is True
    assert result["cooled"] == ["tiny-embed"]
