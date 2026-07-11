# Changelog

## 0.1.0 — 2026-07-11

First working release: full port of the ad-hoc `~/llm-stack` dashboard into an
installable package, per the approved design spec.

### Added
- `hearth` CLI: status, models, warm/cool, pull (`--add`, resume, SHA-256
  verification), logs `-f`, daemon, install/uninstall/up/down/restart,
  doctor, migrate, brain capture/search/reindex.
- `hearthd` daemon on :9300: models/status/chat/config/logs/brain/library
  API + packaged web dashboard (ES modules, no build step).
- Lifecycle engine: TTL auto-unload, follow rules (`app:` and `role:` with a
  sensible chat fallback), crash-loop detection with macOS notification and
  dashboard banner.
- Model library: HF search, verified resumable downloads, fit check,
  one-click **Add to config** (generated ruamel block, roles metadata).
- Brain: sqlite-vec index, incremental reindex, frontmatter-stripped chunks,
  true-cosine scores; `brain` shim delegates to `hearth brain capture`.
- Dashboard: warm-soot Hearthia identity, ambient hearth glow, temperature-
  semantic actions, TTL countdown rings, failure banners, live logs.
- `hearth migrate` adopts an existing `~/llm-stack` in place.

### Notes
- Regression guards: TTL-poisoning (never poll `/upstream/...`), hermetic
  test settings, SSE reconnect, vec_chunks hygiene.
- E2E smoke: `uvx --from playwright python tests/e2e/smoke.py`.
