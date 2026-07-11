"""llama-swap client. All HTTP to the gateway lives here — nowhere else."""

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
        return r.json().get("running", [])

    async def warm(self, model_id: str, timeout: float = 300.0) -> bool:
        try:
            r = await self._client.get(
                f"{self.base_url}/upstream/{model_id}/health", timeout=timeout
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def cool(self, model_id: str | None = None) -> bool:
        path = "/api/models/unload" + (f"/{model_id}" if model_id else "")
        try:
            r = await self._client.post(f"{self.base_url}{path}")
            return r.status_code in (200, 204)
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
