"""
Export exactly what the web page draws, from the same monitor as 03_measure.py.

For fault 14 (reactor cooling-water valve sticking) and fault 15 (condenser cooling-water
valve sticking):
- the representative run: the one whose flag rate after 08:00 is closest to the fault's median
  over runs 1-20, the lowest-numbered on a tie (rule stated on the page);
- its monitor index per sample: max(T²/limit, SPE/limit), so 1.0 = the flag limit;
- the same index for its exact normal twin (fault-free testing run with the same number);
- the first flagged sample and the alarm (third flag in a row) after the fault starts;
- which readings moved most vs the twin after hour 8 (median std ratio over runs 1-20);
- the hourly spread of one trend, fault run vs twin.
Also ASSERTS the twin property for every faulty run used (all 20 faults x runs 1-20): samples
1-160 identical to the fault-free run with the same number. Writes out/page_data.json.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
measure = importlib.import_module("03_measure")
from tags import catalogue  # noqa: E402  pylint: disable=wrong-import-position

TREND = {14: "xmeas_9", 15: "xmeas_22"}
NAMES = {row["column"]: row for row in catalogue()}


def index_series(monitor: "measure.PcaMonitor", data: np.ndarray) -> np.ndarray:
    """Monitor index: the larger of T²/limit and SPE/limit."""
    t2, spe = monitor.score(data)
    return np.maximum(t2 / monitor.t2_limit, spe / monitor.spe_limit)


def spread_ratios(faulty: pd.DataFrame, twins: pd.DataFrame, fault: int) -> dict[str, float]:
    """Median over runs 1-20 of post-onset std(fault run) / std(normal twin), per variable."""
    ratios = {col: [] for col in measure.VARIABLES}
    for run_no, run in faulty[faulty["faultNumber"] == fault].groupby("simulationRun"):
        twin = twins[twins["simulationRun"] == run_no]
        post_f = run[run["sample"] > measure.ONSET_SAMPLE]
        post_t = twin[twin["sample"] > measure.ONSET_SAMPLE]
        for col in measure.VARIABLES:
            ratios[col].append(post_f[col].std(ddof=1) / post_t[col].std(ddof=1))
    return {col: float(np.median(vals)) for col, vals in ratios.items()}


def hourly_spread(fault_vals: np.ndarray, twin_vals: np.ndarray) -> dict:
    """Standard deviation of each 1-hour block (20 samples), fault run vs twin, plus the
    ratio of their means after the fault starts."""
    per_hour = 60 // measure.SAMPLE_MINUTES
    hours = len(fault_vals) // per_hour
    fault_sd = fault_vals[:hours * per_hour].reshape(hours, per_hour).std(axis=1, ddof=1)
    twin_sd = twin_vals[:hours * per_hour].reshape(hours, per_hour).std(axis=1, ddof=1)
    onset_hour = measure.ONSET_SAMPLE // per_hour
    return {"fault": np.round(fault_sd, 4).tolist(), "twin": np.round(twin_sd, 4).tolist(),
            "ratio_after_onset": round(float(fault_sd[onset_hour:].mean() / twin_sd[onset_hour:].mean()), 2)}


def check_twins(faulty: pd.DataFrame, twins: pd.DataFrame) -> int:
    """Every faulty run's samples 1-160 must equal its fault-free twin's. Returns runs checked."""
    cols = ["sample"] + measure.VARIABLES
    pre_twin = {r: g.sort_values("sample")[cols].to_numpy()
                for r, g in twins[twins["sample"] <= measure.ONSET_SAMPLE].groupby("simulationRun")}
    checked = 0
    for (fault, run_no), grp in faulty[faulty["sample"] <= measure.ONSET_SAMPLE].groupby(
            ["faultNumber", "simulationRun"]):
        if not np.array_equal(grp.sort_values("sample")[cols].to_numpy(), pre_twin[run_no]):
            print(f"ERROR: fault {fault} run {run_no} does not match its twin before 08:00")
            sys.exit(1)
        checked += 1
    return checked


def main() -> None:
    """Build and write page_data.json."""
    train = measure.load("fault_free_training.parquet")[measure.VARIABLES].to_numpy(np.float64)
    try:
        monitor = measure.PcaMonitor(train, measure.CPV_TARGET, measure.CONFIDENCE)
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: PCA fit failed - {exc}")
        sys.exit(1)
    faulty = measure.load("faulty_testing_runs1-20.parquet")
    twins = measure.load("fault_free_testing.parquet")
    twins = twins[twins["simulationRun"] <= faulty["simulationRun"].max()]
    try:
        with open(measure.OUT_DIR / "monitor.json", encoding="utf-8") as handle:
            summary = json.load(handle)
        table = pd.read_csv(measure.OUT_DIR / "measurements.csv").set_index("fault")
        sens = pd.read_csv(measure.OUT_DIR / "sensitivity.csv")
    except (OSError, ValueError) as exc:
        print(f"ERROR: run 03_measure.py and 05_sensitivity.py first - {exc}")
        sys.exit(1)

    summary["twin_runs_checked"] = check_twins(faulty, twins)
    print(f"twins: {summary['twin_runs_checked']} faulty runs identical to their twin before 08:00")
    page = {"monitor": summary, "sensitivity": sens.to_dict(orient="records"), "faults": {}}
    for fault, trend_col in TREND.items():
        rates = {}
        for run_no, run in faulty[faulty["faultNumber"] == fault].groupby("simulationRun"):
            post = run[run["sample"] > measure.ONSET_SAMPLE][measure.VARIABLES].to_numpy(np.float64)
            rates[int(run_no)] = float(monitor.flags(post)["any"].mean())
        median = float(np.median(list(rates.values())))
        run_no = min(rates, key=lambda r: (abs(rates[r] - median), r))
        run = faulty[(faulty["faultNumber"] == fault) & (faulty["simulationRun"] == run_no)].sort_values("sample")
        twin = twins[twins["simulationRun"] == run_no].sort_values("sample")
        idx_f = index_series(monitor, run[measure.VARIABLES].to_numpy(np.float64))
        idx_t = index_series(monitor, twin[measure.VARIABLES].to_numpy(np.float64))
        onset_pos = int(np.argmax(run["sample"].to_numpy() > measure.ONSET_SAMPLE))
        flagged = idx_f > 1.0
        hit = measure.alarm_index(flagged, onset_pos, measure.CONSECUTIVE)
        first_flag = onset_pos + int(np.argmax(flagged[onset_pos:])) if flagged[onset_pos:].any() else None
        ratios = spread_ratios(faulty, twins, fault)
        ranked = sorted(((r, c) for c, r in ratios.items()), reverse=True)
        delay = table.loc[fault, "median_alarm_delay_min"]
        page["faults"][str(fault)] = {
            "fault": fault, "type": table.loc[fault, "type"], "disturbed": table.loc[fault, "disturbed"],
            "detection_rate_pct_all_runs": float(table.loc[fault, "detection_rate_any_pct"]),
            "runs_alarmed_of_20": int(table.loc[fault, "runs_alarmed_of_20"]),
            "median_alarm_delay_min": None if pd.isna(delay) else float(delay),
            "run": run_no, "run_rule": "flag rate closest to the median of runs 1-20, lowest-numbered on a tie",
            "run_detection_rate_pct": round(100 * rates[run_no], 2),
            "twin_detection_rate_pct_after_onset": round(100 * float((idx_t[onset_pos:] > 1.0).mean()), 2),
            "first_flag_sample": None if first_flag is None else int(run["sample"].iloc[first_flag]),
            "alarm_sample": None if hit is None else int(run["sample"].iloc[hit]),
            "index_fault": np.round(idx_f, 3).tolist(),
            "index_twin": np.round(idx_t, 3).tolist(),
            "trend": {"tag": NAMES[trend_col]["tag"], "name": NAMES[trend_col]["name"],
                      "unit": NAMES[trend_col]["unit"]},
            "moved_most": [{"tag": NAMES[c]["tag"], "name": NAMES[c]["name"], "column": c,
                            "spread_ratio": round(r, 4)} for r, c in ranked[:5]],
            "spread_ratio_by_column": {c: round(r, 4) for c, r in ratios.items()},
            "spread_hourly": hourly_spread(run[trend_col].to_numpy(np.float64), twin[trend_col].to_numpy(np.float64)),
        }
        info = page["faults"][str(fault)]
        run_hourly = {c: hourly_spread(run[c].to_numpy(np.float64), twin[c].to_numpy(np.float64))["ratio_after_onset"]
                      for c in measure.VARIABLES}
        info["trend_hourly_readings_wider_in_run"] = sum(
            1 for c, r in run_hourly.items() if c != trend_col and r > run_hourly[trend_col])
        print(f"fault {fault}: run {run_no} ({info['run_detection_rate_pct']}% vs twin "
              f"{info['twin_detection_rate_pct_after_onset']}%) | first flag sample {info['first_flag_sample']} "
              f"| alarm sample {info['alarm_sample']} | median alarm delay {info['median_alarm_delay_min']}")
        for row in info["moved_most"]:
            print(f"    {row['spread_ratio']:7.4f}x  {row['tag']:9s} {row['name']}")
    print(f"    fault 15 XMV(11) ratio: {page['faults']['15']['spread_ratio_by_column']['xmv_11']}")
    out = measure.OUT_DIR / "page_data.json"
    try:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(page, handle, separators=(",", ":"))
    except OSError as exc:
        print(f"ERROR: cannot write {out} - {exc}")
        sys.exit(1)
    print(f"[DONE] {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
