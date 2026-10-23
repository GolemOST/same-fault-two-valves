"""
A second, simpler check aimed at valve two: has the condenser loop been jumpier than normal?

How it came about (stated on the page): designed AFTER looking at fault 15 runs 1-20. Comparing
each faulty run with its twin (probe_fault15.py) shows the sticking valve adds fast jitter to the
condenser cooling-water outlet temperature, XMEAS(22). Runs 1-20 therefore cannot test it fairly,
so it is tested on fault 15 runs 21-500, which were not opened until the check was fixed, each
against its own fault-free twin.

Definitions:
- Jitter at sample t: sample variance (ddof 1) of the last WINDOW sample-to-sample changes of
  XMEAS(22). Undefined, and never flagged, until the window is full.
- Limit: the 99th percentile of the jitter over every full window of the 500 fault-free TRAINING
  runs. No fault data is used to set it.
- Flag rate, alarm (3 flags in a row after 08:00, timed at the third) and delay: exactly as in
  03_measure.py.
- Windows: 480 samples (24 h, the check on the page) and 160 samples (8 h, one shift).
Also runs the PCA monitor from 03_measure.py on the same unseen runs, like for like.

Writes out/condenser_check.json. Re-check: pipeline/04_recheck.py.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
measure = importlib.import_module("03_measure")
from tags import FAULTS, catalogue  # noqa: E402  pylint: disable=wrong-import-position

SIGNAL = "xmeas_22"
WINDOWS = (480, 160)
LIMIT_PERCENTILE = 99
HOLDOUT_FILE = "faulty15_testing_runs21-500.parquet"
FIRES_ON_SHARE = 0.5        # "fires on" another fault = alarms in at least half of its runs 1-20
NAMES = {row["column"]: row for row in catalogue()}


def jitter(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling variance of the last `window` sample-to-sample changes; NaN until the window is full."""
    changes = np.diff(values, prepend=np.nan)
    return pd.Series(changes).rolling(window, min_periods=window).var().to_numpy()


def score(frame: pd.DataFrame, flag_fn) -> dict:
    """Flag rate after 08:00 (mean over runs), runs alarmed and median delay, per the 03 definitions."""
    post, delays, alarmed = [], [], 0
    for _run, run in frame.groupby("simulationRun"):
        run = run.sort_values("sample")
        flags = flag_fn(run)
        after = run["sample"].to_numpy() > measure.ONSET_SAMPLE
        post.append(flags[after].mean())
        onset_pos = int(np.argmax(after))
        hit = measure.alarm_index(flags, onset_pos, measure.CONSECUTIVE)
        if hit is not None:
            alarmed += 1
            delays.append((hit - onset_pos + 1) * measure.SAMPLE_MINUTES)
    runs = int(frame["simulationRun"].nunique())
    return {"runs": runs, "flagged_pct_after_onset": round(100 * float(np.mean(post)), 2),
            "runs_alarmed": alarmed, "runs_alarmed_pct": round(100 * alarmed / runs, 1),
            "median_alarm_delay_h": round(float(np.median(delays)) / 60, 1) if delays else None}


def check_twins(faulty: pd.DataFrame, normal: pd.DataFrame) -> int:
    """Every unseen faulty run must equal its fault-free twin before 08:00. Returns runs checked."""
    cols = ["sample"] + measure.VARIABLES
    pre_normal = normal[normal["sample"] <= measure.ONSET_SAMPLE]
    twins = {r: g.sort_values("sample")[cols].to_numpy() for r, g in pre_normal.groupby("simulationRun")}
    checked = 0
    for run_no, grp in faulty[faulty["sample"] <= measure.ONSET_SAMPLE].groupby("simulationRun"):
        if run_no not in twins or not np.array_equal(grp.sort_values("sample")[cols].to_numpy(), twins[run_no]):
            print(f"ERROR: unseen fault 15 run {run_no} does not match its twin before 08:00")
            sys.exit(1)
        checked += 1
    return checked


def main() -> None:
    """Set the limits on training data, then score every set."""
    train = measure.load("fault_free_training.parquet")
    normal = measure.load("fault_free_testing.parquet")
    faulty = measure.load("faulty_testing_runs1-20.parquet")
    holdout = measure.load(HOLDOUT_FILE)
    if holdout["faultNumber"].unique().tolist() != [15]:
        print(f"ERROR: {HOLDOUT_FILE} should hold fault 15 only")
        sys.exit(1)
    held_runs = sorted(holdout["simulationRun"].unique().tolist())
    if held_runs[0] <= faulty["simulationRun"].max():
        print("ERROR: the unseen runs overlap the runs the check was designed on")
        sys.exit(1)
    twins_dev = normal[normal["simulationRun"].isin(faulty["simulationRun"].unique())]
    twins_held = normal[normal["simulationRun"].isin(held_runs)]
    out = {"signal": {"column": SIGNAL, "tag": NAMES[SIGNAL]["tag"], "name": NAMES[SIGNAL]["name"]},
           "limit_percentile": LIMIT_PERCENTILE, "fires_on_share": FIRES_ON_SHARE,
           "design_runs": [int(faulty["simulationRun"].min()), int(faulty["simulationRun"].max())],
           "holdout_runs": [held_runs[0], held_runs[-1]], "holdout_run_count": len(held_runs),
           "holdout_twins_checked": check_twins(holdout, normal), "windows": {}}
    print(f"unseen runs {held_runs[0]}-{held_runs[-1]}: {out['holdout_twins_checked']} match their twin before 08:00")

    for window in WINDOWS:
        train_jitter = np.concatenate([jitter(run.sort_values("sample")[SIGNAL].to_numpy(np.float64), window)
                                       for _r, run in train.groupby("simulationRun")])
        limit = float(np.nanpercentile(train_jitter, LIMIT_PERCENTILE))

        def flag_fn(run: pd.DataFrame, limit: float = limit, window: int = window) -> np.ndarray:
            values = jitter(run[SIGNAL].to_numpy(np.float64), window)
            return np.nan_to_num(values, nan=-np.inf) > limit

        others = {}
        for fault_id, group in faulty.groupby("faultNumber"):
            if int(fault_id) != 15:
                others[str(int(fault_id))] = score(group, flag_fn)["runs_alarmed"]
        per_fault_runs = int(faulty.groupby("faultNumber")["simulationRun"].nunique().min())
        fires = [int(f) for f, n in others.items() if n >= FIRES_ON_SHARE * per_fault_runs]
        result = {
            "window_samples": window, "window_hours": window * measure.SAMPLE_MINUTES // 60, "limit": limit,
            "design": {"fault": score(faulty[faulty["faultNumber"] == 15], flag_fn), "twins": score(twins_dev, flag_fn)},
            "holdout": {"fault": score(holdout, flag_fn), "twins": score(twins_held, flag_fn)},
            "normal": score(normal, flag_fn),
            "other_faults_runs_alarmed": others,
            "fires_on_faults": fires,
            "fires_on_names": {str(f): FAULTS[f][1] for f in fires},
        }
        out["windows"][str(window)] = result
        held = result["holdout"]
        print(f"{result['window_hours']:>2} h window (limit {limit:.4f}): unseen fault runs alarmed "
              f"{held['fault']['runs_alarmed_pct']}% vs twins {held['twins']['runs_alarmed_pct']}% | flagged "
              f"{held['fault']['flagged_pct_after_onset']}% | normal runs alarmed {result['normal']['runs_alarmed']} of "
              f"{result['normal']['runs']} | median delay {held['fault']['median_alarm_delay_h']} h | also fires on {fires}")

    try:
        monitor = measure.PcaMonitor(train[measure.VARIABLES].to_numpy(np.float64), measure.CPV_TARGET,
                                     measure.CONFIDENCE)
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: PCA fit failed - {exc}")
        sys.exit(1)

    def pca_flags(run: pd.DataFrame) -> np.ndarray:
        return monitor.flags(run[measure.VARIABLES].to_numpy(np.float64))["any"]

    out["pca_holdout"] = {"fault": score(holdout, pca_flags), "twins": score(twins_held, pca_flags)}
    pca = out["pca_holdout"]
    print(f"PCA on the same unseen runs: alarmed {pca['fault']['runs_alarmed_pct']}% vs twins "
          f"{pca['twins']['runs_alarmed_pct']}% | flagged {pca['fault']['flagged_pct_after_onset']}% vs "
          f"{pca['twins']['flagged_pct_after_onset']}%")

    target = measure.OUT_DIR / "condenser_check.json"
    try:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
    except OSError as exc:
        print(f"ERROR: cannot write {target} - {exc}")
        sys.exit(1)
    print(f"[DONE] {target}")


if __name__ == "__main__":
    main()
