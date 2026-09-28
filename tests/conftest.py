import os
from pathlib import Path

import pytest

SAMPLE_YAML = """\
# gateway config — comments must survive edits
healthCheckTimeout: 300

macros:
  llama-server: /opt/homebrew/bin/llama-server
  models_dir: /tmp/models

models:
  # the flagship
  "big-coder":
    name: "Big Coder"
    description: "Flagship coding model."
    cmd: |
      ${llama-server}
      --port ${PORT}
      --model ${models_dir}/big.gguf
      --ctx-size 32768
      --temp 0.7
    ttl: 600
    aliases:
      - coder
      - default
    metadata:
      roles: [chat]

  "tiny-embed":
    name: "Tiny Embed"
    description: "Embeddings."
    cmd: |
      ${llama-server}
      --port ${PORT}
      --model ${models_dir}/embed.gguf
      --embeddings
      --ctx-size 8192
    metadata:
      roles: [embed]
"""


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep Settings() away from the developer's real config and directories.

    The config file is written, not merely pointed at, so a default stack_dir
    can never resolve to ~/.hearthia and silently collect test data.
    """
    config = tmp_path / "hearthia-config.toml"
    sandbox = tmp_path / "hearthia-home"
    (sandbox / "models").mkdir(parents=True, exist_ok=True)
    config.write_text(f'[paths]\nstack_dir = "{sandbox}"\nmodels_dir = "{sandbox}/models"\n')
    monkeypatch.setenv("HEARTHIA_CONFIG", str(config))
    for var in [v for v in os.environ if v.startswith("HEARTHIA_") and v != "HEARTHIA_CONFIG"]:
        monkeypatch.delenv(var)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "llama-swap.yaml"
    p.write_text(SAMPLE_YAML)
    return p


@pytest.fixture
def backups_dir(tmp_path: Path) -> Path:
    return tmp_path / "backups"
