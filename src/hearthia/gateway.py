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

    async def running(self) -> list[dict]:
        try:
            r = await self._client.get(f"{self.base_url}/running")
        except httpx.HTTPError:
            return []
        if r.status_code != 200:
            return []
        try:
            data = r.json().get("running") or []
        except ValueError:
            return []
        return data if isinstance(data, list) else []

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

    async def chat_stream(self, body: bytes) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(600.0, connect=300.0),
        ) as r:
            async for chunk in r.aiter_bytes():
                yield chunk
