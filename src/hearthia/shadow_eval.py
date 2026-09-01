"""Shadow-eval health gate.

llama-swap's HTTP health check confirms a model server started and answers
`/health` — not that it produces coherent output. A model warmed with a
mismatched chat template, a broken quantization, or a `--ctx-size` that
overflows its trained context can pass that health check and still return
garbage or an empty completion on its very first real request. No
local-model runtime fires a real inference request before calling a warm
"healthy" — this module does, with one minimal, cheap canary prompt.
"""

import logging

import httpx

from hearthia.gateway import Gateway

log = logging.getLogger("hearthia.shadow_eval")

_CANARY_PROMPT = "Reply with exactly one word: OK"
_DEFAULT_MAX_TOKENS = 8
_DEFAULT_TIMEOUT = 30.0


async def canary_check(
    gw: Gateway,
    model_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Send one minimal completion to confirm a warmed model actually produces
    output, not just that its HTTP health check passed.

    Returns ``{"ok": bool, "text": str}`` on a real response (``ok`` is False
    for an empty/whitespace-only completion, the most common broken-warm
    symptom), or ``{"ok": False, "error": str}`` when the request itself
    failed or timed out.
    """
    try:
        response = await gw.chat(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": _CANARY_PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
    except (httpx.HTTPError, TypeError, ValueError) as e:
        log.warning("shadow-eval canary failed for %s: %s", model_id, e)
        return {"ok": False, "error": str(e)}

    choices = response.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
    ok = bool(text.strip())
    if not ok:
        log.warning("shadow-eval canary for %s returned an empty completion", model_id)
    return {"ok": ok, "text": text.strip()[:200]}
