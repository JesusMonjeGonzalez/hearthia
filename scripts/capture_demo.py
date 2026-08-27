#!/usr/bin/env python3
"""Capture an animated GIF of the Hearthia demo dashboard.

Run the demo, warm models through the demo gateway, and screenshot the real
UI so the README shows the actual product. Regenerate with:

    uvx --from playwright --with pillow python scripts/capture_demo.py
"""

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn  # noqa: E402

from hearthia.demo import create_demo_app  # noqa: E402

PORT = 9317
OUT = Path(__file__).parent.parent / "docs" / "assets" / "hearthia-demo.gif"
FRAMES: list[Path] = []


def serve() -> None:
    uvicorn.run(create_demo_app(port=PORT), host="127.0.0.1", port=PORT, log_level="error")


async def capture() -> None:
    import httpx
    from playwright.async_api import async_playwright

    base = f"http://127.0.0.1:{PORT}"
    for _ in range(50):  # wait for the demo daemon to accept connections
        try:
            async with httpx.AsyncClient() as c:
                if (await c.get(f"{base}/api/status")).status_code == 200:
                    break
        except httpx.HTTPError:
            await asyncio.sleep(0.2)
    tmp = Path("/tmp/hearthia-gif-frames")
    tmp.mkdir(exist_ok=True)
    shot = 0

    async def snap(page, name: str, delay: float = 0.0) -> None:
        nonlocal shot
        await asyncio.sleep(delay)
        await page.screenshot(path=str(tmp / f"{shot:02d}-{name}.png"))
        FRAMES.append(tmp / f"{shot:02d}-{name}.png")
        shot += 1

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1280, "height": 820}, device_scale_factor=2
        )
        await page.goto(base)
        await page.wait_for_selector("#model-cards .card")
        await snap(page, "idle", 1.0)

        # warm the flagship and the notes model: the memory map comes alive
        await page.evaluate("fetch('/api/models/qwen-coder-30b/load', {method:'POST'})")
        await page.wait_for_timeout(2600)
        await page.evaluate("fetch('/api/models/gemma-notes-12b/load', {method:'POST'})")
        await page.wait_for_timeout(2600)
        await snap(page, "two-warm", 1.2)

        # chat: the canned reply streams through the real pipeline
        await page.click('[data-tab="chat"]')
        await page.fill("#chat-input", "What am I looking at?")
        await page.click("#chat-send")
        await page.wait_for_timeout(1500)
        await snap(page, "chat-streaming")

        await page.wait_for_timeout(6000)
        await snap(page, "chat-done")

        await page.click('[data-tab="models"]')
        await page.wait_for_timeout(800)
        await snap(page, "models-warm")

        await browser.close()


def assemble() -> None:
    from PIL import Image

    frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in FRAMES]
    # halve resolution: crisp on retina, light on disk
    frames = [f.resize((f.width // 2, f.height // 2), Image.LANCZOS) for f in frames]
    durations = [1600, 900, 900, 1400, 2600, 1400][: len(frames)]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, {len(frames)} frames)")


if __name__ == "__main__":
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    asyncio.run(capture())
    assemble()
