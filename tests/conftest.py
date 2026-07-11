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


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "llama-swap.yaml"
    p.write_text(SAMPLE_YAML)
    return p


@pytest.fixture
def backups_dir(tmp_path: Path) -> Path:
    return tmp_path / "backups"
