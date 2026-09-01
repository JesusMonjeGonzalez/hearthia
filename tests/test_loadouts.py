from hearthia.loadouts import (
    defined_loadouts,
    get_loadout,
    loadout_cool,
    loadout_load,
    loadout_plan,
    loadouts_affected_by_drift,
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


async def test_loadout_cool_preserves_members_shared_with_another_loadout(tmp_path):
    s = _settings(
        tmp_path,
        {
            "coding": LoadoutSettings(models=["big-coder", "tiny-embed"]),
            "notes": LoadoutSettings(models=["tiny-embed"]),
        },
    )
    gw = StubGateway(
        running=[
            {"model": "big-coder", "state": "ready"},
            {"model": "tiny-embed", "state": "ready"},
        ]
    )

    result = await loadout_cool(s, gw, _reg(tmp_path), "coding")

    assert result["ok"] is True
    assert result["cooled"] == ["big-coder"]
    assert result["preserved_shared"] == [{"model": "tiny-embed", "loadouts": ["notes"]}]
    assert gw.cooled == ["big-coder"]


def test_loadouts_affected_by_drift_only_returns_containing_loadouts(tmp_path):
    s = _settings(
        tmp_path,
        {
            "coding": LoadoutSettings(models=["big-coder", "tiny-embed"]),
            "notes": LoadoutSettings(models=["tiny-embed"]),
        },
    )
    affected = loadouts_affected_by_drift(s, _reg(tmp_path), "big-coder")
    assert [a["loadout"] for a in affected] == ["coding"]
    assert "fits" in affected[0]
    assert "total_bytes" in affected[0]


def test_loadouts_affected_by_drift_ignores_model_not_in_any_loadout(tmp_path):
    s = _settings(tmp_path, _loadouts(["big-coder"]))
    assert loadouts_affected_by_drift(s, _reg(tmp_path), "tiny-embed") == []


def test_loadouts_affected_by_drift_reports_no_longer_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr("hearthia.budget.wired_limit_bytes", lambda total: 1)
    s = _settings(tmp_path, _loadouts(["big-coder", "tiny-embed"]))
    affected = loadouts_affected_by_drift(s, _reg(tmp_path), "big-coder")
    assert affected == [
        {
            "loadout": "coding",
            "fits": False,
            "total_bytes": affected[0]["total_bytes"],
            "wired_limit": 1,
        }
    ]
