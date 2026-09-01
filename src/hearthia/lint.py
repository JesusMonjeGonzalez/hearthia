"""Config lint: sanity-checks llama-swap.yaml against real GGUF headers.

llama-swap itself does not validate `--ctx-size` against a model's trained
context, alias collisions across models, or loadout/lifecycle rules
pointing at ids that no longer exist — each only surfaces as a confusing
runtime failure (a warm that mysteriously behaves badly, a loadout that
"warms nothing", a lifecycle follower that never spawns). No local-model
runtime lints its config against the actual model headers it manages.
"""

from dataclasses import dataclass

from hearthia.budget import profile_for
from hearthia.registry import Model, Registry
from hearthia.settings import LoadoutSettings, Settings

_ROPE_FLAGS = ("--rope-scale", "--rope-scaling", "--yarn-orig-ctx", "--yarn-ext-factor")


@dataclass(frozen=True)
class LintIssue:
    severity: str  # "warn" | "info"
    message: str


def _check_ctx_vs_trained(model: Model) -> LintIssue | None:
    if not model.ctx:
        return None
    profile = profile_for(model)
    if profile is None or not profile.context_length:
        return None
    if model.ctx <= profile.context_length:
        return None
    if any(flag in model.cmd for flag in _ROPE_FLAGS):
        return None  # RoPE scaling can legitimately extend past the trained context
    return LintIssue(
        "warn",
        f"{model.id}: --ctx-size {model.ctx:,} exceeds the model's trained context "
        f"{profile.context_length:,} tokens (no RoPE scaling flag set) — quality may "
        "degrade past its trained length",
    )


def _check_missing_file(model: Model) -> LintIssue | None:
    if model.file is None:
        return LintIssue("warn", f"{model.id}: no --model path found in its cmd")
    if not model.file.exists():
        return LintIssue("warn", f"{model.id}: weights file not found at {model.file}")
    return None


def _check_ttl_missing(model: Model, rules: dict[str, str]) -> LintIssue | None:
    if model.ttl or model.id in rules or model.embedding:
        return None
    return LintIssue("info", f"{model.id}: no ttl and no lifecycle rule — never auto-unloads")


def _check_alias_collisions(models: list[Model]) -> list[LintIssue]:
    seen: dict[str, str] = {}
    issues: list[LintIssue] = []
    for m in models:
        for name in (m.id, *m.aliases):
            owner = seen.get(name)
            if owner is None:
                seen[name] = m.id
            elif owner != m.id:
                issues.append(
                    LintIssue(
                        "warn",
                        f"alias '{name}' is used by both '{owner}' and '{m.id}' — "
                        "llama-swap routing for it is ambiguous",
                    )
                )
    return issues


def _check_lifecycle_rules(models: list[Model], rules: dict[str, str]) -> list[LintIssue]:
    known = {m.id for m in models}
    issues: list[LintIssue] = []
    for follower_id, rule in rules.items():
        if follower_id not in known:
            issues.append(LintIssue("warn", f"[lifecycle] '{follower_id}' is not a model"))
        if ":" not in rule:
            issues.append(
                LintIssue(
                    "warn",
                    f"[lifecycle] '{follower_id}' rule {rule!r} is malformed "
                    "(expected 'kind:target')",
                )
            )
    return issues


def _check_loadout_members(
    models: list[Model], loadouts: dict[str, LoadoutSettings] | None
) -> list[LintIssue]:
    known = {m.id for m in models}
    issues: list[LintIssue] = []
    for name, cfg in (loadouts or {}).items():
        for mid in cfg.models:
            if mid not in known:
                issues.append(
                    LintIssue("warn", f"[loadouts.{name}] references unknown model '{mid}'")
                )
    return issues


def lint(settings: Settings, registry: Registry) -> list[LintIssue]:
    """Every check, in a stable order: per-model checks first, then
    cross-model ones. Pure — reads the config and GGUF headers, changes
    nothing."""
    models = registry.models()
    issues: list[LintIssue] = []
    for m in models:
        for issue in (
            _check_ctx_vs_trained(m),
            _check_missing_file(m),
            _check_ttl_missing(m, settings.lifecycle),
        ):
            if issue is not None:
                issues.append(issue)
    issues.extend(_check_alias_collisions(models))
    issues.extend(_check_lifecycle_rules(models, settings.lifecycle))
    issues.extend(_check_loadout_members(models, settings.loadouts))
    return issues
