"""
Three more checks for valve two (his ask 10-23: try to do better than the 05b condenser check).

Designed like 05b: every setting was chosen on fault 15 runs 1-20 and their twins, with limits from the 500
fault-free TRAINING runs only, BEFORE the unseen runs 21-500 were scored. The design comparison (window lengths
tried, one setting kept per idea) is recorded in the README of this lab; this step only runs the kept settings.
  MULTI  four linked readings together: the four that moved most on runs 1-20 (out/page_data.json moved_most:
         XMEAS 22, 11, 16, 18). Each one's log-variance of sample-to-sample changes over the last 8 h, z-scored
         against the training windows; the four summed and divided by 2 (= sqrt 4).
  CUSUM  an accumulating alarm on XMEAS(22): z = change^2 / (mean change^2 in training) - 1;
         S = max(0, S + z - k); k = half the mean z of the design runs after 08:00.
  ALL52  all 52 readings: log-variance of their changes over the last 12 h -> PCA (90% of variance) on the
         training windows -> T2 and SPE, each against its 99th percentile; the statistic is the larger ratio.
Limits: the 99th percentile of each statistic over the training runs. Flag, alarm (3 in a row after 08:00) and
delay exactly as in 03_measure.py / 05b_condenser_check.py.

Writes out/more_checks.json. Re-check: pipeline/04_recheck.py.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
measure = importlib.import_module("03_measure")
check05b = importlib.import_module("05b_condenser_check")

PCT = 99
MULTI_WINDOW, ALL52_WINDOW, ALL52_CPV, CUSUM_K_SHARE = 160, 240, 0.90, 0.5
NAMES = {"multi": "Four linked readings, 8 h", "cusum": "Accumulating alarm (CUSUM) on one reading",
         "all52": "All 52 readings, window statistics, 12 h"}


def diffs(x: np.ndarray) -> np.ndarray:
    """sample-to-sample changes, first row NaN (same convention as 05b)."""
    return np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)])


def rollvar(x: np.ndarray, w: int) -> np.ndarray:
    """variance (ddof 1) of the last w rows, from running sums; NaN until the window is full of changes."""
    out = np.full(x.shape, np.nan)
    if x.shape[0] <= w:
        return out
    v = np.nan_to_num(x)
    c1 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(v, axis=0)])
    c2 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(v * v, axis=0)])
    s1, s2 = c1[w + 1:] - c1[1:-w], c2[w + 1:] - c2[1:-w]
    out[w:] = (s2 - s1 * s1 / w) / (w - 1)
    return out


def logvar(x: np.ndarray, w: int) -> np.ndarray:
    """log of rollvar, floored at 1e-12: a reading that does not move at all for a whole window (it happens only
    in other faults' runs, e.g. fault 6 trips the simulated plant and its readings freeze) gives a large finite
    value instead of -inf. NaN (window not yet full) stays NaN."""
    return np.log(np.maximum(rollvar(x, w), 1e-12))


def arrays(frame: pd.DataFrame) -> list[np.ndarray]:
    return [g.sort_values("sample")[measure.VARIABLES].to_numpy(np.float64) for _, g in frame.groupby("simulationRun")]


def score(stats: list[np.ndarray], limit: float) -> dict:
    """flag rate after 08:00 (mean over runs), runs alarmed, median delay (h) - the 05b definitions."""
    post, delays, alarmed = [], [], 0
    for s in stats:
        flags = np.nan_to_num(s, nan=-np.inf) > limit
        post.append(flags[measure.ONSET_SAMPLE:].mean())
        hit = measure.alarm_index(flags, measure.ONSET_SAMPLE, measure.CONSECUTIVE)
        if hit is not None:
            alarmed += 1
            delays.append((hit - measure.ONSET_SAMPLE + 1) * measure.SAMPLE_MINUTES)
    n = len(stats)
    return {"runs": n, "flagged_pct_after_onset": round(100 * float(np.mean(post)), 2), "runs_alarmed": alarmed,
            "runs_alarmed_pct": round(100 * alarmed / n, 1),
            "median_alarm_delay_h": round(float(np.median(delays)) / 60, 1) if delays else None}


def build_checks(train: list[np.ndarray], design_fault: list[np.ndarray], multi_cols: list[str]) -> dict:
    """Fit each check on the training runs (+ the design runs for CUSUM's k). Returns {key: (stat_fn, limit, settings)}."""
    idx = {c: i for i, c in enumerate(measure.VARIABLES)}
    checks = {}
    # MULTI
    mi = [idx[c] for c in multi_cols]
    lv = np.vstack([logvar(diffs(x[:, mi]), MULTI_WINDOW) for x in train])
    mu, sd = np.nanmean(lv, axis=0), np.nanstd(lv, axis=0)

    def multi(x, mi=mi, mu=mu, sd=sd):
        return ((logvar(diffs(x[:, mi]), MULTI_WINDOW) - mu) / sd).sum(axis=1) / np.sqrt(len(mi))
    checks["multi"] = (multi, {"readings": multi_cols, "window_samples": MULTI_WINDOW})
    # CUSUM
    j = idx["xmeas_22"]
    s2 = float(np.mean(np.concatenate([np.diff(x[:, j]) ** 2 for x in train])))
    excess = float(np.mean(np.concatenate([np.diff(x[:, j])[measure.ONSET_SAMPLE:] ** 2 / s2 - 1 for x in design_fault])))
    k = CUSUM_K_SHARE * excess

    def cusum(x, j=j, s2=s2, k=k):
        z = np.diff(x[:, j]) ** 2 / s2 - 1
        out, acc = np.full(x.shape[0], np.nan), 0.0
        for i, zi in enumerate(z, start=1):
            acc = max(0.0, acc + zi - k)
            out[i] = acc
        return out
    checks["cusum"] = (cusum, {"reading": "xmeas_22", "normal_mean_change_sq": s2, "design_excess": excess, "k": k})
    # ALL52
    feats = [logvar(diffs(x), ALL52_WINDOW) for x in train]
    a = np.vstack([f[~np.isnan(f).any(axis=1)] for f in feats])
    mu52, sd52 = a.mean(axis=0), a.std(axis=0, ddof=1)
    z = (a - mu52) / sd52
    ev, evec = np.linalg.eigh(np.cov(z, rowvar=False))
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    npc = int(np.searchsorted(np.cumsum(ev) / ev.sum(), ALL52_CPV) + 1)
    pm = evec[:, :npc]

    def t2spe(f):
        zf = (f - mu52) / sd52
        t = zf @ pm
        return np.sum(t ** 2 / ev[:npc], axis=1), np.sum((zf - t @ pm.T) ** 2, axis=1)
    t2a, spa = t2spe(a)
    lt, ls = float(np.percentile(t2a, PCT)), float(np.percentile(spa, PCT))

    def all52(x):
        t2, sp = t2spe(logvar(diffs(x), ALL52_WINDOW))
        return np.maximum(t2 / lt, sp / ls)
    checks["all52"] = (all52, {"window_samples": ALL52_WINDOW, "components": npc, "t2_limit": lt, "spe_limit": ls})
    return checks


def main() -> None:
    """Fit on training + design runs, then score design, unseen, normal and other faults."""
    try:
        with open(measure.OUT_DIR / "page_data.json", encoding="utf-8") as handle:
            moved = [r["column"] for r in json.load(handle)["faults"]["15"]["moved_most"][:4]]
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: run 06_export_page.py first ({exc})")
        sys.exit(1)
    train = arrays(measure.load("fault_free_training.parquet"))
    normal_df = measure.load("fault_free_testing.parquet")
    faulty_df = measure.load("faulty_testing_runs1-20.parquet")
    held_df = measure.load(check05b.HOLDOUT_FILE)
    held_runs = sorted(held_df["simulationRun"].unique().tolist())
    design_fault = arrays(faulty_df[faulty_df["faultNumber"] == 15])
    sets = {"design_fault": design_fault, "design_twins": arrays(normal_df[normal_df["simulationRun"] <= 20]),
            "holdout_fault": arrays(held_df), "holdout_twins": arrays(normal_df[normal_df["simulationRun"].isin(held_runs)]),
            "normal": arrays(normal_df)}
    others = {int(f): arrays(g) for f, g in faulty_df.groupby("faultNumber") if int(f) != 15}
    checks = build_checks(train, design_fault, moved)
    out = {"limit_percentile": PCT, "design_runs": [1, 20], "holdout_runs": [held_runs[0], held_runs[-1]], "checks": {}}
    for key, (fn, settings) in checks.items():
        limit = float(np.nanpercentile(np.concatenate([fn(x) for x in train]), PCT))
        res = {"name": NAMES[key], "settings": settings, "limit": limit}
        for name, runs in sets.items():
            res[name] = score([fn(x) for x in runs], limit)
        res["other_faults_runs_alarmed"] = {str(f): score([fn(x) for x in runs], limit)["runs_alarmed"] for f, runs in others.items()}
        res["fires_on_faults"] = [f for f, n in sorted(((int(a), b) for a, b in res["other_faults_runs_alarmed"].items())) if n >= 10]
        out["checks"][key] = res
        h, t, nn = res["holdout_fault"], res["holdout_twins"], res["normal"]
        print(f"{NAMES[key]:44s} unseen faulty {h['runs_alarmed']}/{h['runs']} ({h['runs_alarmed_pct']}%) flagged "
              f"{h['flagged_pct_after_onset']}% delay {h['median_alarm_delay_h']} h | twins {t['runs_alarmed']}/{t['runs']} "
              f"({t['runs_alarmed_pct']}%) | normal {nn['runs_alarmed']}/{nn['runs']}, flagged {nn['flagged_pct_after_onset']}%")
    target = measure.OUT_DIR / "more_checks.json"
    try:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
    except OSError as exc:
        print(f"ERROR: cannot write {target} - {exc}")
        sys.exit(1)
    print(f"[DONE] {target}")


if __name__ == "__main__":
    main()
