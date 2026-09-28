"""Exercise real log controls against a demo daemon, with synthetic log traffic.

uvx --from playwright python tests/e2e/logs.py http://127.0.0.1:9381
"""

import sys

from playwright.sync_api import expect, sync_playwright

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9381"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route(
        "**/api/logs/stream",
        lambda route: route.fulfill(
            content_type="text/plain", body="ready\n[ERROR] synthetic failure\nrecovered\n"
        ),
    )
    page.goto(base)
    page.locator('.tab[data-tab="logs"]').click()
    view = page.locator("#log-view")
    expect(view).to_contain_text("synthetic failure")
    page.get_by_role("button", name="Pause view", exact=True).click()
    expect(page.locator("#logs-pause")).to_have_attribute("aria-pressed", "true")
    page.get_by_role("searchbox", name="Filter log lines").fill("  error  ")
    expect(view).to_have_text("[ERROR] synthetic failure")
    with page.expect_download() as download:
        page.get_by_role("button", name="Export view", exact=True).click()
    assert download.value.suggested_filename.startswith("hearthia-logs-")
    with open(download.value.path(), encoding="utf-8") as exported:
        assert exported.read() == "[ERROR] synthetic failure"
    page.get_by_role("button", name="Clear view", exact=True).click()
    expect(view).to_have_text("")
    page.get_by_role("button", name="Resume", exact=True).click()
    expect(page.locator("#logs-pause")).to_have_attribute("aria-pressed", "false")
    page.get_by_role("searchbox", name="Filter log lines").fill("")
    expect(view).to_contain_text("recovered", timeout=6000)
    page.set_viewport_size({"width": 390, "height": 844})
    for selector in ["#logs-search", "#logs-pause", "#logs-export", "#logs-clear"]:
        box = page.locator(selector).bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
    assert not errors, errors
    browser.close()

print("PASS: logs filter, pause/resume, clear, download and mobile layout")
