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


# Other studies of the same simulator, AS REPORTED in their papers (not measured here), best first. Valve two
# (fault 15) and valve one (fault 14) as % of samples flagged after the fault (= 100 - their missed-detection
# rate where that is what they give), and % flagged on normal data. Text cells are kept exactly as the paper
# gives them; f15_num is only for the build's sentence checks (None = no percentage given). Source in "ref".
# 10-23 v3: the five open-table studies first used were NOT the best published; his question "is there a better
# result?" found Agarwal 2020, Shang & Chen 2018, Du & Du 2018 and Galicia 2012, added here.
MITPDF = "https://web.mit.edu/braatzgroup/"
LITERATURE = [
    {"study": "Agarwal and colleagues (2020)", "url": "https://arxiv.org/abs/2012.03861", "method": "Deep neural network (LSTM)",
     "f15": "100%", "f15_num": 100.0, "normal": "2.41% (all faults)", "f14": 100.0, "ref": "Tables 4-5, §4",
     "note": "Trained on labelled examples of the fault, and nudges a setpoint every 4 h so it shows"},
    {"study": "Shang and Chen (2018)", "url": "https://search.napstic.cn/literature/periodical/0220180600226670",
     "method": "Dynamic transformed-component statistics",
     "f15": "over 95%", "f15_num": 95.0, "normal": "not in the abstract", "f14": None, "ref": "abstract",
     "note": "Figure from the paper's abstract"},
    {"study": "Du and Du (2018)", "url": "https://skoge.folk.ntnu.no/prost/proceedings/ADCHEM-2018/media/files/0146.pdf",
     "method": "Filtered signals + accumulating alarm (CUSUM)", "f15": "detected, after 30 h on average", "f15_num": None,
     "normal": "not given", "f14": None, "ref": "Table 2", "note": "Their own simulation (6-minute samples)"},
    {"study": "Zhao (2017)", "url": "https://arxiv.org/abs/1712.04118", "method": "PCA (SPE)", "f15": "25.6%", "f15_num": 25.6,
     "normal": "16.9%", "f14": 100.0, "ref": "Table II", "note": "Many false alarms"},
    {"study": "Lu and colleagues (2018)", "url": MITPDF + "Lu_JProcessControl_2018.pdf", "method": "Sparse canonical variate analysis",
     "f15": "25.5%", "f15_num": 25.5, "normal": "not given", "f14": 100.0, "ref": "Table 2", "note": ""},
    {"study": "Chiang and Braatz (2003)", "url": MITPDF + "54_Process_monitoring_using_causal_map_and_multivariate_statistics_"
     "fault_detection_and_identification.pdf", "method": "Causal map + statistics", "f15": "24.9%", "f15_num": 24.9,
     "normal": "4.62%", "f14": 100.0, "ref": "Table 2", "note": "Calls this fault unobservable"},
    {"study": "Sun and colleagues (2020)", "url": "https://arxiv.org/abs/1911.04386", "method": "PCA (12 components)",
     "f15": "10.62%", "f15_num": 10.62, "normal": "5.00%", "f14": 100.0, "ref": "Table 1", "note": ""},
    {"study": "Sun and colleagues (2020)", "url": "https://arxiv.org/abs/1911.04386", "method": "Bayesian recurrent neural network",
     "f15": "7.12%", "f15_num": 7.12, "normal": "4.75%", "f14": 100.0, "ref": "Table 1", "note": ""},
    {"study": "Peng, Zhao and Hu (2021)", "url": "https://arxiv.org/abs/2103.07303", "method": "PCA", "f15": "5.75%",
     "f15_num": 5.75, "normal": "1.88%", "f14": 99.87, "ref": "Table II", "note": ""},
    {"study": "Zhao (2017)", "url": "https://arxiv.org/abs/1712.04118", "method": "PCA (T²)", "f15": "4.9%", "f15_num": 4.9,
     "normal": "0.00%", "f14": 99.87, "ref": "Table II", "note": ""},
    {"study": "Galicia, He and Wang (2012)", "url": "https://skoge.folk.ntnu.no/prost/proceedings/cpc8-focapo-2012/data/papers/055.pdf",
     "method": "Window statistics (SPA)", "f15": "not detected", "f15_num": 0.0, "normal": "n/a", "f14": 99.5, "ref": "Table 3 + text",
     "note": "Their own simulation; PCA and dynamic PCA missed it too"},
]
LIT_F14 = [r["f14"] for r in LITERATURE if r["f14"] is not None]


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


def read_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict, str]:
    """Load the pipeline outputs and the template, exiting with what to run if missing."""
    try:
        with open(OUT_DIR / "page_data.json", encoding="utf-8") as handle:
            data = json.load(handle)
        with open(OUT_DIR / "condenser_check.json", encoding="utf-8") as handle:
            check = json.load(handle)
        with open(OUT_DIR / "more_checks.json", encoding="utf-8") as handle:
            more = json.load(handle)
        faults = pd.read_csv(OUT_DIR / "measurements.csv")
        sens = pd.read_csv(OUT_DIR / "sensitivity.csv")
        template = (PAGE_DIR / "template.html").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: missing input - run 03, 05, 05b, 05c and 06 first ({exc})")
        sys.exit(1)
    return data, faults, sens, check, more, template


WHOSE = {"published": "Published study", "textbook": "Textbook method, run here", "mine": "<strong>Mine</strong>"}
MORE_NAMES = {"cusum": "One reading, accumulating alarm (CUSUM)", "multi": "Four readings' jitter, 8&nbsp;h",
              "all52": "All 52 readings' jitter, 12&nbsp;h"}


def my_checks(check: dict, more: dict) -> list[dict]:
    """Every check of mine, scored on the unseen runs: name, unseen faulty runs, their twins, the 500 normal runs."""
    rows = []
    for key in ("480", "160"):
        w = check["windows"][key]
        rows.append({"key": f"day{key}", "name": f"One reading's jitter, {w['window_hours']}&nbsp;h",
                     "fault": w["holdout"]["fault"], "twins": w["holdout"]["twins"], "normal": w["normal"]})
    for key in ("cusum", "multi", "all52"):
        c = more["checks"][key]
        rows.append({"key": key, "name": MORE_NAMES[key], "fault": c["holdout_fault"], "twins": c["holdout_twins"],
                     "normal": c["normal"]})
    return rows


def lit_rows(mon: dict, f15: dict, check: dict, more: dict) -> str:
    """HTML rows for 08: the published studies (as reported, best first), then this page's rows (measured)."""
    html = []
    for r in LITERATURE:
        html.append(f'                <tr><td><a href="{r["url"]}" rel="noopener">{r["study"]}</a></td><td>{WHOSE["published"]}</td>'
                    f'<td>{r["method"]}</td><td class="r">{r["f15"]}</td><td class="r">{r["normal"]}</td><td>{r["note"]}</td></tr>')
    html.append(f'                <tr><td><strong>This page</strong>, 20 runs</td><td>{WHOSE["textbook"]}</td>'
                f'<td>PCA ({mon["components"]} components)</td><td class="r">{f15["detection_rate_pct_all_runs"]:.2f}%</td>'
                f'<td class="r">{mon["normal_flag_pct_after_onset"]:.2f}%</td><td>No different from a normal day</td></tr>')
    for c in my_checks(check, more):
        delay = c["fault"]["median_alarm_delay_h"]
        html.append(f'                <tr class="hl"><td><strong>This page</strong>, {c["fault"]["runs"]} unseen runs</td>'
                    f'<td>{WHOSE["mine"]}</td><td>{c["name"]}</td><td class="r">{c["fault"]["flagged_pct_after_onset"]:.2f}%</td>'
                    f'<td class="r">{c["normal"]["flagged_pct_after_onset"]:.2f}%</td>'
                    f'<td>Typical alarm after {delay:.0f}&nbsp;h</td></tr>')
    return "\n".join(html)


def check_rows(mon: dict, check: dict, more: dict) -> str:
    """HTML rows for 09-10: the textbook monitor and every check of mine, on the same unseen runs."""
    pca = check["pca_holdout"]
    entries = [("Standard monitor (PCA)", WHOSE["textbook"], pca["fault"], pca["twins"],
                mon["normal_runs_with_alarm_after_onset"], "no better than its twins")]
    for c in my_checks(check, more):
        entries.append((c["name"], WHOSE["mine"], c["fault"], c["twins"], c["normal"]["runs_alarmed"],
                        f'{c["fault"]["median_alarm_delay_h"]:.0f}&nbsp;h'))
    rows = []
    for name, whose, fault, twins, normal_runs, when in entries:
        rows.append(f'                <tr><td>{name}</td><td>{whose}</td>'
                    f'<td class="r">{fault["runs_alarmed"]} of {fault["runs"]} ({fault["runs_alarmed_pct"]:.0f}%)</td>'
                    f'<td class="r">{twins["runs_alarmed"]} of {twins["runs"]} ({twins["runs_alarmed_pct"]:.0f}%)</td>'
                    f'<td class="r">{normal_runs} of {mon["normal_runs"]}</td>'
                    f'<td class="r">{fault["flagged_pct_after_onset"]:.1f}%</td><td class="r">{when}</td></tr>')
    return "\n".join(rows)


def and_list(items: list) -> str:
    """[2, 12, 17] -> '2, 12 and 17'."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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
    data, faults, sens, check, more, html = read_inputs()
    f14, f15, mon = data["faults"]["14"], data["faults"]["15"], data["monitor"]
    day, shift, pca_held = check["windows"]["480"], check["windows"]["160"], check["pca_holdout"]
    cu, mu, a52 = (more["checks"][k] for k in ("cusum", "multi", "all52"))
    mine = my_checks(check, more)
    plain_pca = [r for r in LITERATURE if r["method"].startswith("PCA") and r["f15_num"] is not None]
    built_for_it = [r for r in LITERATURE if r["study"].startswith(("Agarwal", "Shang"))]

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
        "LIT_ROWS": lit_rows(mon, f15, check, more), "CHECK_ROWS": check_rows(mon, check, more),
        "LIT_F14_LOW": f"{min(LIT_F14):.1f}", "LIT_F14_HIGH": f"{max(LIT_F14):g}", "LIT_N": str(len({r['study'] for r in LITERATURE})),
        "CU_HO_FAULT_PCT": f"{cu['holdout_fault']['runs_alarmed_pct']:.0f}",
        "CU_HO_TWIN_PCT": f"{cu['holdout_twins']['runs_alarmed_pct']:.0f}",
        "CU_HO_FLAG": f"{cu['holdout_fault']['flagged_pct_after_onset']:.1f}",
        "CU_HO_DELAY": f"{cu['holdout_fault']['median_alarm_delay_h']:.0f}",
        "CU_NORMAL_RUNS": str(cu["normal"]["runs_alarmed"]), "CU_DEV_TWINS": str(cu["design_twins"]["runs_alarmed"]),
        "MU_OTHER_N": str(len(mu["fires_on_faults"])), "A52_OTHER_N": str(len(a52["fires_on_faults"])),
        "A52_HO_FAULT_PCT": f"{a52['holdout_fault']['runs_alarmed_pct']:.0f}",
        "A52_HO_TWIN_PCT": f"{a52['holdout_twins']['runs_alarmed_pct']:.0f}",
        "OTHER_FAULTS": str(len(mu["other_faults_runs_alarmed"])),
        "CC_TAG": check["signal"]["tag"], "CC_HOURS": str(day["window_hours"]),
        "CC_SHIFT_HOURS": str(shift["window_hours"]), "CC_PCTL": str(check["limit_percentile"]),
        "CC_DESIGN_LAST": str(check["design_runs"][1]),
        "CC_HELD_FIRST": str(check["holdout_runs"][0]), "CC_HELD_LAST": str(check["holdout_runs"][1]),
        "CC_HELD_RUNS": str(check["holdout_run_count"]),
        "CC_DEV_FAULT": str(day["design"]["fault"]["runs_alarmed"]),
        "CC_DEV_TWINS": str(day["design"]["twins"]["runs_alarmed"]),
        "CC_HO_FAULT_PCT": f"{day['holdout']['fault']['runs_alarmed_pct']:.0f}",
        "CC_HO_TWIN_PCT": f"{day['holdout']['twins']['runs_alarmed_pct']:.0f}",
        "CC_HO_FLAG": f"{day['holdout']['fault']['flagged_pct_after_onset']:.2f}",
        "CC_HO_DELAY": f"{day['holdout']['fault']['median_alarm_delay_h']:.0f}",
        "CC_NORMAL_RUNS": str(day["normal"]["runs_alarmed"]),
        "PCA_NORMAL_RUNS_AFTER": str(mon["normal_runs_with_alarm_after_onset"]),
        "PCA_HO_FAULT_PCT": f"{pca_held['fault']['runs_alarmed_pct']:.0f}",
        "PCA_HO_TWIN_PCT": f"{pca_held['twins']['runs_alarmed_pct']:.0f}",
        "CC_SHIFT_DELAY": f"{shift['holdout']['fault']['median_alarm_delay_h']:.0f}",
        "CC_SHIFT_TWIN_PCT": f"{shift['holdout']['twins']['runs_alarmed_pct']:.0f}",
        "CC_OTHER_N": str(len(day["fires_on_faults"])), "CC_OTHER_LIST": and_list(day["fires_on_faults"]),
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
        (min(LIT_F14) >= 99.5 and f14["detection_rate_pct_all_runs"] >= min(LIT_F14),
         "08: on valve one every study that reports it catches 99.5-100%, this page included"),
        (all(r["f15_num"] - float(r["normal"].rstrip("%")) <= 11 for r in plain_pca),
         "08: plain PCA stays close to its own normal-data rate"),
        (min(r["f15_num"] for r in built_for_it) >= 95, "08: methods built for it reach 95-100%"),
        (any(r["f15"] == "not detected" for r in LITERATURE), "08: window statistics (SPA) did not detect it"),
        (day["holdout"]["fault"]["runs_alarmed_pct"] >= 80 and day["holdout"]["twins"]["runs_alarmed_pct"] <= 15,
         "09: the 24 h check separates unseen faulty runs from their twins"),
        (pca_held["fault"]["runs_alarmed_pct"] - pca_held["twins"]["runs_alarmed_pct"] < 10,
         "09: PCA barely separates the same unseen runs"),
        (day["normal"]["runs_alarmed"] < mon["normal_runs_with_alarm_after_onset"],
         "09: fewer false alarms on normal running than the PCA monitor"),
        (max(r["f15_num"] for r in plain_pca) < day["holdout"]["fault"]["flagged_pct_after_onset"]
         < min(r["f15_num"] for r in built_for_it),
         "09: its flag rate beats the plain monitors in 08, not the methods built for this fault"),
        (cu["holdout_fault"]["flagged_pct_after_onset"] == max(c["fault"]["flagged_pct_after_onset"] for c in mine)
         and cu["holdout_fault"]["runs_alarmed_pct"] == max(c["fault"]["runs_alarmed_pct"] for c in mine),
         "10: the accumulating alarm catches the most of my checks"),
        (cu["holdout_fault"]["median_alarm_delay_h"] < day["holdout"]["fault"]["median_alarm_delay_h"],
         "10: the accumulating alarm alarms sooner than the one-day check"),
        (cu["holdout_twins"]["runs_alarmed_pct"] > 2 * day["holdout"]["twins"]["runs_alarmed_pct"]
         and cu["normal"]["runs_alarmed"] > day["normal"]["runs_alarmed"],
         "10: its price is about twice the false alarms of the one-day check"),
        (cu["design_twins"]["runs_alarmed"] <= 3, "10: on the 20 design runs that price did not show"),
        (abs(mu["holdout_fault"]["runs_alarmed_pct"] - day["holdout"]["fault"]["runs_alarmed_pct"]) < 2
         and mu["holdout_twins"]["runs_alarmed_pct"] > day["holdout"]["twins"]["runs_alarmed_pct"],
         "10: four readings caught about as many as one, with more false alarms"),
        (a52["holdout_fault"]["runs_alarmed_pct"] - a52["holdout_twins"]["runs_alarmed_pct"] < 15,
         "10: all 52 readings barely separate faulty runs from twins"),
        (min(len(mu["fires_on_faults"]), len(a52["fires_on_faults"])) >= len(mu["other_faults_runs_alarmed"]) / 2,
         "10: four and 52 readings alarm on most of the other faults"),
        (max(c["fault"]["flagged_pct_after_onset"] for c in mine) < min(r["f15_num"] for r in built_for_it),
         "10: none of mine reaches the methods built for this fault"),
        (12 <= day["holdout"]["fault"]["median_alarm_delay_h"] <= 30, "09: it needs most of a day"),
        ({12, 18} <= set(day["fires_on_faults"]), "09: it also fires on faults 12 and 18 (the condenser)"),
        (shift["holdout"]["fault"]["median_alarm_delay_h"] < day["holdout"]["fault"]["median_alarm_delay_h"]
         and shift["normal"]["runs_alarmed"] > day["normal"]["runs_alarmed"]
         and shift["holdout"]["twins"]["runs_alarmed_pct"] > day["holdout"]["twins"]["runs_alarmed_pct"],
         "09: the 8 h version is quicker but raises more false alarms"),
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
