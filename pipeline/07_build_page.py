"""
Build the lab page from page/template.html + out/*.json/csv. Every number in the copy comes
from the measured data, never typed by hand.

    python pipeline/07_build_page.py [--date YYYY-MM-DD]

Writes:
- page/index.html                                  the page
- page/preview/lab/same-fault-two-valves/index.html + styles.css/app.js copied from the website
  folder, so a local server rooted at page/preview/ resolves /styles.css like production does.
  Skipped (with a notice) when there is no website folder, as in a fresh clone.

Refuses to build (exit 1) if a sentence the copy makes is no longer true of the data, if the
article text has an em dash, or if any {{PLACEHOLDER}} is left unfilled. An optional
pipeline/local_config.json (not in the repo) adds a list of words the copy must not contain and
the path to the website folder.
"""

import argparse
import datetime as dt
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

LAB_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = LAB_DIR / "out"
PAGE_DIR = LAB_DIR / "page"
LOCAL_CONFIG = Path(__file__).resolve().parent / "local_config.json"
SLUG = "same-fault-two-valves"
SAMPLE_MINUTES = 3
ONSET_SAMPLE = 160

# FICTION, written for the experiment (the simulator has no people). Labelled as fiction on the page.
NOTES = {
    "14": {"sample": 168, "text": "Reactor CW valve. Panel called it in. Stem not moving smooth: "
                                  "holds, then jumps. Reactor temp swinging with it. Called maintenance."},
    "15": {"sample": 202, "text": "Condenser CW valve sticking. Stem holds, then jumps. "
                                  "Panel says normal. Called maintenance."},
}


def load_local_config() -> dict:
    """The optional local config: banned copy words and the website folder. {} when absent."""
    if not LOCAL_CONFIG.exists():
        print(f"NOTICE: no {LOCAL_CONFIG.name} - building without the local word sweep or the site preview")
        return {}
    try:
        with open(LOCAL_CONFIG, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {LOCAL_CONFIG} exists but cannot be read - fix or remove it ({exc})")
        sys.exit(1)
    site = config.get("site_dir")
    config["site_dir"] = (LOCAL_CONFIG.parent / site).resolve() if site else None
    return config


def read_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    """Load the pipeline outputs and the template, exiting with what to run if missing."""
    try:
        with open(OUT_DIR / "page_data.json", encoding="utf-8") as handle:
            data = json.load(handle)
        faults = pd.read_csv(OUT_DIR / "measurements.csv")
        sens = pd.read_csv(OUT_DIR / "sensitivity.csv")
        template = (PAGE_DIR / "template.html").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: missing input - run 03, 05 and 06 first ({exc})")
        sys.exit(1)
    return data, faults, sens, template


def clock(sample: int) -> str:
    """Sample number (1 = minute 3) -> HH:MM on day 1."""
    minutes = sample * SAMPLE_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def css_version(site_dir: Path | None) -> str:
    """The ?v= token the live site currently uses for styles.css ('dev' with no website folder)."""
    if site_dir is None:
        return "dev"
    try:
        html = (site_dir / "index.html").read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {site_dir / 'index.html'} for the CSS version - {exc}")
        sys.exit(1)
    match = re.search(r"styles\.css\?v=([0-9a-z]+)", html)
    if not match:
        print("ERROR: no styles.css?v= token in the site's index.html")
        sys.exit(1)
    return match.group(1)


def fault_rows(faults: pd.DataFrame) -> str:
    """HTML rows for the all-faults table; faults 14 and 15 highlighted."""
    rows = []
    for rec in faults.itertuples():
        delay = "none" if pd.isna(rec.median_alarm_delay_min) else f"{rec.median_alarm_delay_min:g} min"
        cls = ' class="hl"' if rec.fault in (14, 15) else ""
        rows.append(f'                <tr{cls}><td>{rec.fault}</td><td>{rec.type}</td><td>{rec.disturbed}</td>'
                    f'<td class="r">{rec.detection_rate_any_pct:.2f}%</td><td class="r">{rec.runs_alarmed_of_20} of 20</td>'
                    f'<td class="r">{delay}</td></tr>')
    return "\n".join(rows)


def sens_rows(sens: pd.DataFrame) -> str:
    """HTML rows for the tuning sweep."""
    return "\n".join(
        f'                <tr><td class="r">{r.variance_target:.0%}</td><td class="r">{r.confidence * 100:g}%</td>'
        f'<td class="r">{r.components}</td><td class="r">{r.normal_day_flag_pct:.2f}%</td>'
        f'<td class="r">{r.fault_14_flag_pct:.2f}%</td><td class="r">{r.fault_15_flag_pct:.2f}%</td></tr>'
        for r in sens.itertuples())


def sweep(html: str, config: dict) -> list[str]:
    """Return every copy rule the page breaks."""
    problems = []
    lower = html.lower()
    for phrase in config.get("ro_phrases", []):
        if phrase in lower:
            problems.append(f"banned phrase: '{phrase}'")
    text_only = re.sub(r"<[^>]+>", " ", html.split('<script type="application/json"')[0]).lower()
    for word in config.get("narrowing", []):
        if re.search(rf"\b{re.escape(word)}\b", text_only):
            problems.append(f"banned word: '{word}'")
    body = html.split('<div class="body">', 1)[-1].split('<nav class="artnav"', 1)[0]
    if "—" in body or "–" in body:
        problems.append("em/en dash in the article text")
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", html)
    if left:
        problems.append(f"unfilled placeholders: {sorted(set(left))}")
    return problems


def main() -> None:
    """Fill, sweep, write, stage the preview."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()
    config = load_local_config()
    site_dir = config.get("site_dir")
    data, faults, sens, html = read_inputs()
    f14, f15, mon = data["faults"]["14"], data["faults"]["15"], data["monitor"]

    notes = {k: {"sample": v["sample"], "time": clock(v["sample"])} for k, v in NOTES.items()}
    data["notes"] = notes
    rest = f14["moved_most"][3]["spread_ratio"]
    values = {
        "PUBLISH_DATE": args.date, "PUBLISH_DATE_DOTS": args.date.replace("-", "·"),
        "CSS_VERSION": css_version(site_dir),
        "TWIN_RUNS_CHECKED": str(mon["twin_runs_checked"]),
        "NORMAL_FLAG": f"{mon['normal_flag_pct']['any']:.2f}",
        "NORMAL_FLAG_AFTER": f"{mon['normal_flag_pct_after_onset']:.2f}",
        "NORMAL_RUNS_SUSTAINED": str(mon["normal_runs_with_sustained_alarm"]),
        "NORMAL_RUNS": str(mon["normal_runs"]),
        "TWINS_ALARMED": str(mon["twin_runs_1_20_with_alarm_after_onset"]),
        "COMPONENTS": str(mon["components"]), "CPV": f"{100 * mon['cpv']:.1f}",
        "F14_RUN": str(f14["run"]), "F15_RUN": str(f15["run"]),
        "F14_ALARM_DELAY": f"{f14['median_alarm_delay_min']:g}",
        "F14_ALARM_TIME": clock(f14["alarm_sample"]),
        "F14_RATE": f"{f14['detection_rate_pct_all_runs']:.2f}",
        "F14_M1": f"{f14['moved_most'][0]['spread_ratio']:.1f}",
        "F14_M2": f"{f14['moved_most'][1]['spread_ratio']:.1f}",
        "F14_M3": f"{f14['moved_most'][2]['spread_ratio']:.1f}",
        "F14_REST_PCT": str(math.ceil(round((rest - 1) * 100, 6))),
        "F15_RATE": f"{f15['detection_rate_pct_all_runs']:.2f}",
        "F15_RUNS_ALARMED": str(f15["runs_alarmed_of_20"]),
        "F15_RUN_RATE": f"{f15['run_detection_rate_pct']:.2f}",
        "F15_TWIN_RATE": f"{f15['twin_detection_rate_pct_after_onset']:.2f}",
        "F15_M1": f"{f15['moved_most'][0]['spread_ratio']:.2f}",
        "F15_XMV11_RATIO": f"{f15['spread_ratio_by_column']['xmv_11']:.2f}",
        "F15_TREND_UNIT": f15["trend"]["unit"],
        "F15_RUN_SPREAD": f"{f15['spread_hourly']['ratio_after_onset']:.2f}",
        "NOTE14_TIME": notes["14"]["time"], "NOTE14_TEXT": NOTES["14"]["text"],
        "NOTE15_TIME": notes["15"]["time"], "NOTE15_TEXT": NOTES["15"]["text"],
        "FAULT_ROWS": fault_rows(faults), "SENS_ROWS": sens_rows(sens),
    }
    # each sentence the copy makes, as a check on the data; fail rather than ship a false line
    claims = [
        (f14["moved_most"][0]["tag"] == "XMV(10)" and f15["moved_most"][0]["tag"] == "XMEAS(22)",
         "top movers are the reactor CW valve (fault 14) and condenser CW outlet temp (fault 15)"),
        (f14["first_flag_sample"] == ONSET_SAMPLE + 1, "valve one: the first sample after 08:00 is already flagged"),
        (f14["runs_alarmed_of_20"] == 20, "valve one: all 20 runs alarmed"),
        (abs(f15["detection_rate_pct_all_runs"] - mon["normal_flag_pct_after_onset"]) < 0.5,
         "valve two flags 'about as often as a normal day' (within 0.5 points)"),
        (f15["spread_ratio_by_column"]["xmv_11"] < 1.02, "XMV(11) 'barely changes'"),
        (1 <= f15["trend_hourly_readings_wider_in_run"] <= 8, "chart C: 'a few other values widen about as much'"),
        (f15["run_detection_rate_pct"] - f15["twin_detection_rate_pct_after_onset"] < 0.5,
         "table: run indistinguishable from its twin"),
    ]
    broken = [text for ok, text in claims if not ok]
    if broken:
        print("ERROR: the data no longer supports these sentences - rewrite the copy:\n  " + "\n  ".join(broken))
        sys.exit(1)
    for key, val in values.items():
        html = html.replace("{{" + key + "}}", val)
    html = html.replace("{{DATA_JSON}}", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))

    problems = sweep(html, config)
    if problems:
        print("ERROR: copy sweep failed:\n  " + "\n  ".join(problems))
        sys.exit(1)

    target = PAGE_DIR / "index.html"
    preview = PAGE_DIR / "preview"
    try:
        target.write_text(html, encoding="utf-8")
        if site_dir is not None:
            (preview / "lab" / SLUG).mkdir(parents=True, exist_ok=True)
            (preview / "lab" / SLUG / "index.html").write_text(html, encoding="utf-8")
            for name in ("styles.css", "app.js"):
                shutil.copyfile(site_dir / name, preview / name)
    except OSError as exc:
        print(f"ERROR: cannot write the page or preview - {exc}")
        sys.exit(1)
    staged = f"preview root {preview}" if site_dir is not None else "no site preview (no website folder)"
    print(f"[DONE] {target} ({target.stat().st_size / 1024:.0f} KB) | sweep clean | "
          f"notes at {notes['14']['time']} / {notes['15']['time']} | {staged}")


if __name__ == "__main__":
    main()
