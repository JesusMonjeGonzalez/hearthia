"""llama-swap client. All HTTP to the gateway lives here — nowhere else."""

import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx


class Gateway:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9292",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def is_up(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def inventory(self) -> list[dict] | None:
        """Resident models, or ``None`` when the inventory could not be read.

        The distinction matters for the memory gate: an unreachable gateway is
        not the same as an empty machine, and treating it as one lets a load
        through on top of models that are still resident.
        """
        try:
            r = await self._client.get(f"{self.base_url}/running")
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json().get("running") or []
        except ValueError:
            return None
        return data if isinstance(data, list) else None

    async def running(self) -> list[dict]:
        """Resident models for display; an unreadable inventory reads as empty."""
        return await self.inventory() or []

    async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
        try:
            r = await self._client.get(
                f"{self.base_url}/upstream/{quote(model_id, safe='')}/health", timeout=timeout
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def cool(self, model_id: str | None = None) -> bool:
        path = "/api/models/unload" + (f"/{quote(model_id, safe='')}" if model_id else "")
        try:
            r = await self._client.post(f"{self.base_url}{path}")
            return r.status_code in (200, 204)
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    async def metrics(self) -> str:
        try:
            r = await self._client.get(f"{self.base_url}/metrics")
            return r.text if r.status_code == 200 else ""
        except httpx.HTTPError:
            return ""

    async def events(self) -> AsyncIterator[dict]:
        async with self._client.stream(
            "GET",
            f"{self.base_url}/api/events",
            timeout=httpx.Timeout(None, connect=5.0),
        ) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    yield json.loads(line[5:])
                except json.JSONDecodeError:
                    continue

    async def logs_stream(self) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "GET",
            f"{self.base_url}/logs/stream",
            timeout=httpx.Timeout(None, connect=5.0),
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk

    async def chat(self, body: dict, timeout: float = 600.0) -> dict:
        """Non-streaming chat completion. Returns the full JSON response."""
        r = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout, connect=min(300.0, timeout)),
        )
        r.raise_for_status()
        return r.json()

    def _endpoint(self, path: str, model: str | None) -> str:
        # llama-swap does not proxy /tokenize or /apply-template at the root,
        # but does expose them under /upstream/<model>/…; the direct path is
        # kept for stacks that front llama-server without a proxy.
        return f"{self.base_url}/upstream/{model}/{path}" if model else f"{self.base_url}/{path}"

    async def tokenize(self, text: str, model: str | None = None) -> int | None:
        """Exact token count for ``text``, or None when the server cannot say.

        Never raises: a missing endpoint or an unreachable server falls back
        to the byte estimate in ``context_budget`` instead of failing a turn.
        """
        try:
            r = await self._client.post(
                self._endpoint("tokenize", model),
                json={"content": text},
                timeout=httpx.Timeout(15.0, connect=2.0),
            )
            if r.status_code != 200:
                return None
            tokens = r.json().get("tokens")
            return len(tokens) if isinstance(tokens, list) else None
        except Exception:  # noqa: BLE001 — optional measurement, never fatal
            return None

    async def apply_template(self, messages: list[dict], model: str | None = None) -> str | None:
        """Render messages with the model's own chat template, when supported."""
        try:
            r = await self._client.post(
                self._endpoint("apply-template", model),
                json={"messages": messages},
                timeout=httpx.Timeout(15.0, connect=2.0),
            )
            if r.status_code != 200:
                return None
            prompt = r.json().get("prompt")
            return prompt if isinstance(prompt, str) else None
        except Exception:  # noqa: BLE001 — optional measurement, never fatal
            return None

    async def ping(self, model: str) -> bool:
        """One-token completion through the proxy, to refresh llama-swap's TTL.

        Proxied requests are what reset the activity timer; a health probe may
        not count. With ``cache_prompt`` the cost is a cache hit plus a single
        decoded token, so it is cheap enough to run while long tools execute.
        """
        try:
            r = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "keepalive"}],
                    "max_tokens": 1,
                    "stream": False,
                    "cache_prompt": True,
                },
                timeout=httpx.Timeout(30.0, connect=2.0),
            )
            return r.status_code == 200
        except Exception:  # noqa: BLE001 — a failed ping must never fail a turn
            return False

    async def chat_once(self, body: bytes) -> dict:
        """One non-streaming completion, used by subagents.

        Simpler than SSE parsing for a nested loop, and it still carries
        ``timings``/``usage`` so subagent inference is accounted honestly.
        """
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(600.0, connect=300.0),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("gateway returned a non-object completion")
        return data

    async def chat_stream(self, body: bytes) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(600.0, connect=300.0),
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk
