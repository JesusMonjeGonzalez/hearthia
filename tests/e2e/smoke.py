"""Hearthia dashboard smoke test — run against a live daemon.

Not part of the pytest suite (needs playwright + a running hearthd):

    uvx --from playwright python tests/e2e/smoke.py [http://127.0.0.1:9300]

Checks every tab renders without console errors and that the chat module is
actually wired (the bug class pytest can't see: a module nobody imports).
"""

import json
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9300"
TABS = ["models", "chat", "brain", "library", "config", "logs", "treepact"]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    status = json.load(urllib.request.urlopen(f"{BASE}/api/status", timeout=5))
    for key in ("swap_up", "health", "running", "system"):
        if key not in status:
            fail(f"/api/status missing '{key}'")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE, wait_until="networkidle")
        for tab in TABS:
            page.click(f'.tab[data-tab="{tab}"]')
            page.wait_for_timeout(300)
            if not page.is_visible(f"#tab-{tab}"):
                fail(f"tab '{tab}' did not activate")

        # chat module wiring: New chat must create a conversation item
        # (this is pure frontend — no model request is made)
        page.click('.tab[data-tab="chat"]')
        before = page.locator(".conv-item").count()
        page.click("#conv-new")
        page.wait_for_timeout(200)
        if page.locator(".conv-item").count() != before + 1:
            fail("chat module not wired — '#conv-new' created no conversation")

        # narrow viewport: the conversation list must be reachable via the
        # drawer toggle instead of permanently display:none (regression for
        # the bug the roadmap flagged).
        page.set_viewport_size({"width": 500, "height": 800})
        page.wait_for_timeout(200)
        if page.locator(".conv-list").is_visible():
            fail("conv-list should start closed on narrow viewports")
        page.click("#conv-toggle")
        page.wait_for_timeout(200)
        if not page.locator(".conv-list").is_visible():
            fail("#conv-toggle did not open the conversation drawer")
        page.click(".conv-item")
        page.wait_for_timeout(200)
        if page.locator(".conv-list").is_visible():
            fail("selecting a conversation should close the drawer on narrow viewports")
        page.set_viewport_size({"width": 1280, "height": 900})

        # TreePact panel must stay strictly read-only: no run/resume/cancel/
        # cleanup affordance should ever exist in the dashboard.
        page.click('.tab[data-tab="treepact"]')
        page.wait_for_timeout(500)
        mutable = page.locator(
            "#tab-treepact button:not(#treepact-refresh):not(.treepact-row), "
            "#tab-treepact [data-action='run'], #tab-treepact [data-action='resume'], "
            "#tab-treepact [data-action='cancel'], #tab-treepact [data-action='cleanup']"
        ).count()
        if mutable:
            fail(f"TreePact panel exposes {mutable} unexpected control(s) beyond Refresh")

        if errors:
            fail(f"console errors: {errors}")
        browser.close()

    print(f"PASS: {len(TABS)} tabs, chat wired, no console errors ({BASE})")


if __name__ == "__main__":
    main()
