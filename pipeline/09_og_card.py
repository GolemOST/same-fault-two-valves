"""
Render the page's own share image (1200x630) from out/page_data.json with a private headless
Chromium. Both valves' monitor index on the SAME 0-20 scale (every 4th sample, no smoothing),
so the thumbnail cannot exaggerate the difference.

    python pipeline/09_og_card.py   ->  page/og-card.png (+ preview copy)
"""

import json
import sys
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright is not installed - python -m pip install playwright")
    sys.exit(1)

LAB_DIR = Path(__file__).resolve().parent.parent
OUT = LAB_DIR / "page" / "og-card.png"
PREVIEW = LAB_DIR / "page" / "preview" / "lab" / "same-fault-two-valves" / "og-card.png"
W, H = 1200, 630
PANEL_W, PANEL_H, Y_MAX, EVERY = 500, 190, 20.0, 4


def path(values: list[float]) -> str:
    """SVG path for one series in a PANEL_W x PANEL_H box, 0..Y_MAX, every EVERY-th sample."""
    pts = values[EVERY - 1::EVERY]
    step = PANEL_W / (len(pts) - 1)
    return "".join(f"{'L' if i else 'M'}{i * step:.1f},{PANEL_H * (1 - min(v, Y_MAX) / Y_MAX):.1f}"
                   for i, v in enumerate(pts))


def panel(x: int, title: str, verdict: str, fault: dict) -> str:
    """One mini chart with its alarm line, onset tick and labels."""
    alarm_y = PANEL_H * (1 - 1 / Y_MAX)
    onset_x = PANEL_W * 8 / 48
    return f"""
    <g transform="translate({x},330)">
      <text x="0" y="-22" class="lab">{title}</text>
      <line x1="0" x2="{PANEL_W}" y1="{PANEL_H}" y2="{PANEL_H}" class="base"/>
      <line x1="{onset_x:.1f}" x2="{onset_x:.1f}" y1="0" y2="{PANEL_H}" class="onset"/>
      <path d="{path(fault['index_fault'])}" class="run"/>
      <path d="{path(fault['index_twin'])}" class="twin"/>
      <line x1="0" x2="{PANEL_W}" y1="{alarm_y:.1f}" y2="{alarm_y:.1f}" class="alarm"/>
      <text x="0" y="{PANEL_H + 40}" class="verdict">{verdict}</text>
    </g>"""


def main() -> None:
    """Build the card HTML and screenshot it."""
    try:
        with open(LAB_DIR / "out" / "page_data.json", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read out/page_data.json - run 06_export_page.py first ({exc})")
        sys.exit(1)
    f14, f15 = data["faults"]["14"], data["faults"]["15"]
    delay = f"{f14['median_alarm_delay_min']:g}"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      html,body{{margin:0;width:{W}px;height:{H}px;background:#0B1220;overflow:hidden}}
      svg text{{font-family:'Cascadia Code',Consolas,ui-monospace,monospace}}
      .kick{{fill:#C8961E;font-size:19px;letter-spacing:2.4px}}
      .title{{fill:#E9E4D8;font-family:'Palatino Linotype',Palatino,Georgia,serif!important;font-size:76px;font-weight:600}}
      .sub{{fill:#9AA5B5;font-family:'Palatino Linotype',Palatino,Georgia,serif!important;font-size:28px;font-style:italic}}
      .lab{{fill:#9AA5B5;font-size:18px;letter-spacing:1.5px}}
      .verdict{{fill:#E9E4D8;font-size:22px}}
      .foot{{fill:#7A8699;font-size:17px;letter-spacing:1px}}
      .base{{stroke:rgba(122,134,153,.35);stroke-width:1}}
      .onset{{stroke:#7A8699;stroke-width:1;stroke-dasharray:2 4}}
      .alarm{{stroke:#E9E4D8;stroke-opacity:.55;stroke-width:1.2}}
      .run{{stroke:#3987e5;stroke-width:2;fill:none;stroke-linejoin:round}}
      .twin{{stroke:#199e70;stroke-width:1.5;fill:none;stroke-dasharray:5 4}}
    </style></head><body>
    <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
      <rect x="70" y="58" width="300" height="3" fill="#C8961E"/>
      <text x="70" y="96" class="kick">LAB 01 · AN EXPERIMENT ON A PUBLIC BENCHMARK</text>
      <text x="70" y="188" class="title">Same Fault, Two Valves</text>
      <text x="70" y="240" class="sub">Two sticking cooling-water valves. One standard monitor.</text>
      {panel(70, "VALVE ONE · REACTOR", f"Alarm within {delay} minutes", f14)}
      {panel(630, "VALVE TWO · CONDENSER", "No different from a normal day", f15)}
      <text x="{W - 70}" y="{H - 34}" text-anchor="end" class="foot">raymondcapisinio.com</text>
      <text x="70" y="{H - 34}" class="foot">Same scale · dashed = same day, no fault</text>
    </svg></body></html>"""
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            page.set_content(html)
            page.screenshot(path=str(OUT))
            browser.close()
        PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        PREVIEW.write_bytes(OUT.read_bytes())
    except (PlaywrightError, OSError) as exc:
        print(f"ERROR: could not render {OUT} - {exc}")
        sys.exit(1)
    print(f"[DONE] {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
