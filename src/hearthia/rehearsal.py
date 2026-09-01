"""Fleet health rehearsal.

A model can sit cold for weeks between uses; the first sign it is actually
broken — a bad quant, a stale symlink, a chat-template mismatch — is often
the moment a user genuinely needs it. `hearth rehearse` warms each
currently-cold registered model just long enough to fire one shadow-eval
canary (`shadow_eval.py`), then cools it back down if it was not already
warm: an explicit, manual health check across the whole roster, never a
background scheduler nobody asked for. No local-model runtime offers this.
"""

import logging

from hearthia.budget import plan_warm_now
from hearthia.gateway import Gateway
from hearthia.registry import Model
from hearthia.settings import Settings
from hearthia.shadow_eval import canary_check

log = logging.getLogger("hearthia.rehearsal")


async def rehearse(
    s: Settings, gw: Gateway, targets: list[Model], all_models: list[Model] | None = None
) -> list[dict]:
    """Warm, canary-check, and (if it was cold) cool every model in ``targets``.

    ``all_models`` is the full registry, used only for the budget gate's
    resident-total accounting when it differs from ``targets`` (rehearsing a
    subset must still see every other model actually running); defaults to
    ``targets`` when not given.

    Returns one result dict per model:
    ``{"model_id", "status", "detail"}`` where ``status`` is one of
    ``"ok"``, ``"canary_failed"``, ``"blocked"``, ``"warm_failed"``. A model
    already warm is canary-checked in place and never cooled — rehearsal
    must never disturb a model actually in use.
    """
    all_models = targets if all_models is None else all_models
    results: list[dict] = []
    for model in targets:
        running = await gw.running()
        running_ids = {m.get("model", "") for m in running}
        was_warm = model.id in running_ids

        if not was_warm:
            decision = plan_warm_now(
                all_models, model.id, running, mode=s.memory.mode if s.memory else "enforce"
            )
            if not decision.allowed:
                results.append(
                    {"model_id": model.id, "status": "blocked", "detail": decision.blocked_reason}
                )
                continue
            if not await gw.warm(model.id, timeout=s.gateway.health_timeout):
                results.append(
                    {"model_id": model.id, "status": "warm_failed", "detail": "gateway warm failed"}
                )
                continue

        canary = await canary_check(gw, model.id)
        if canary.get("ok"):
            results.append({"model_id": model.id, "status": "ok", "detail": canary.get("text")})
        else:
            results.append(
                {
                    "model_id": model.id,
                    "status": "canary_failed",
                    "detail": canary.get("error") or "empty completion",
                }
            )

        if not was_warm:
            await gw.cool(model.id)
    return results
