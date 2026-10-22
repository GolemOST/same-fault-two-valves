"""
Render the staged page with its own headless Chromium (not a shared browser session) and
screenshot it: desktop 1280 and phone 390, dark and light. Also reports console errors,
horizontal overflow, chart count, and whether each chart's tooltip responds to a hover.

    python pipeline/08_render_check.py [--url http://127.0.0.1:8765/lab/same-fault-two-valves/]
Screenshots -> page/shots/.
"""

import argparse
import sys
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright is not installed - python -m pip install playwright")
    sys.exit(1)

LAB_DIR = Path(__file__).resolve().parent.parent
SHOTS = LAB_DIR / "page" / "shots"
VIEWS = [("desktop", 1280, 900), ("phone", 390, 844)]
THEMES = ["dark", "light"]


def main() -> None:
    """Shoot every view x theme; print the checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/lab/same-fault-two-valves/")
    args = parser.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch()
            for view, width, height in VIEWS:
                for theme in THEMES:
                    context = browser.new_context(viewport={"width": width, "height": height},
                                                  color_scheme=theme, device_scale_factor=2 if view == "phone" else 1)
                    page = context.new_page()
                    errors: list[str] = []
                    page.on("console", lambda m, e=errors: e.append(m.text) if m.type == "error" else None)
                    page.on("pageerror", lambda exc, e=errors: e.append(str(exc)))
                    page.goto(args.url, wait_until="networkidle")
                    title = page.title()
                    charts = page.locator("svg.lc").count()
                    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                    tips = []
                    for chart in ("a", "b", "c"):
                        hit = page.locator(f'[data-chart="{chart}"] rect.lc-hit')
                        hit.scroll_into_view_if_needed()
                        box = hit.bounding_box()
                        page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)
                        tips.append(page.locator(f'[data-chart="{chart}"] .lc-tip.on').count() == 1)
                    page.mouse.move(0, 0)
                    # the site's scroll-reveal (app.js) keeps blocks at opacity 0 until they enter
                    # the viewport; walk the page like a reader so the full-page shot shows them
                    # wheel steps, like a reader: scrollTo() jumps fight the site's smooth scrolling
                    # and skipped blocks (a false FAIL on phone, debugged 2026-10-22)
                    page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
                    page.wait_for_timeout(300)
                    for _step in range(400):
                        page.mouse.wheel(0, 300)
                        page.wait_for_timeout(60)
                        if page.evaluate("scrollY + innerHeight >= document.documentElement.scrollHeight - 2"):
                            break
                    page.wait_for_timeout(900)
                    hidden = page.evaluate("[...document.querySelectorAll('[data-reveal]')]"
                                           ".filter(e => !e.classList.contains('in'))"
                                           ".map(e => e.className + ' h=' + e.offsetHeight + ' top=' + "
                                           "Math.round(e.getBoundingClientRect().top + scrollY) + ' \"' + "
                                           "e.textContent.trim().slice(0, 30) + '\"')")
                    if hidden:
                        errors.append(f"never revealed: {hidden}")
                    page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
                    shot = SHOTS / f"{view}-{theme}.png"
                    page.screenshot(path=str(shot), full_page=True)
                    for chart in ("a", "b", "c"):
                        page.locator(f'[data-chart="{chart}"]').screenshot(path=str(SHOTS / f"chart-{chart}-{view}-{theme}.png"))
                    ok = title.startswith("Same Fault") and charts == 3 and overflow <= 0 and all(tips) and not errors
                    failed |= not ok
                    print(f"{'PASS' if ok else 'FAIL'} {view:7s} {theme:5s} | title='{title}' | charts {charts} "
                          f"| h-overflow {overflow}px | tooltips {tips} | console errors {errors[:3]}")
                    context.close()
            browser.close()
    except PlaywrightError as exc:
        print(f"ERROR: Playwright failed - {exc}")
        sys.exit(1)
    print(f"[DONE] screenshots in {SHOTS}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
