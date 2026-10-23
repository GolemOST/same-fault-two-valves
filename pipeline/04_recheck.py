"""
Independent re-check of EVERY measured number the page quotes (out/monitor.json,
out/measurements.csv, out/page_data.json, out/condenser_check.json).

Same definitions as 03_measure.py / 05b_condenser_check.py / 06_export_page.py, different code
path: PCA by SVD of the data matrix (not eigh of the covariance), limits re-coded from the
formulas, alarms found with rolling windows (not a loop), rates and ratios with pandas group
operations, twins checked with a merge, the condenser jitter from running sums (not pandas
rolling) and its limit by np.quantile. Must agree within tolerance, or exit 1. Run AFTER
06_export_page.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tags import catalogue

LAB_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = LAB_DIR / "data" / "parquet"
OUT_DIR = LAB_DIR / "out"
COLUMNS = [row["column"] for row in catalogue()]
ONSET, STEP_MIN, IN_A_ROW = 160, 3, 3
PCT_TOL = 0.006      # reported percentages are rounded to 0.01
RATIO_TOL = 0.0006   # ratios rounded to 4 dp; the hourly ratio is rounded to 2 dp, own tolerance below


def read(path: Path) -> pd.DataFrame:
    """Parquet or CSV reader that exits loudly."""
    try:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {path} - {exc}")
        sys.exit(1)


class Recheck:
    """Holds the independently fitted monitor and the comparison results."""

    def __init__(self) -> None:
        try:
            with open(OUT_DIR / "monitor.json", encoding="utf-8") as handle:
                self.mon = json.load(handle)
            with open(OUT_DIR / "page_data.json", encoding="utf-8") as handle:
                self.page = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"ERROR: run 03_measure.py and 06_export_page.py first ({exc})")
            sys.exit(1)
        self.table = read(OUT_DIR / "measurements.csv").set_index("fault")
        self.failed = False
        train = read(PARQUET_DIR / "fault_free_training.parquet")[COLUMNS].to_numpy(np.float64)
        self.centre, self.spread = train.mean(axis=0), train.std(axis=0, ddof=1)
        z_train = (train - self.centre) / self.spread
        rows = z_train.shape[0]
        _u, sing, vt = np.linalg.svd(z_train, full_matrices=False)
        self.var = sing ** 2 / (rows - 1)
        self.keep = int(np.argmax(self.var.cumsum() / self.var.sum() >= self.mon["cpv_target"])) + 1
        self.basis = vt[:self.keep].T
        conf = self.mon["confidence"]
        self.t2_lim = self.keep * (rows ** 2 - 1) / (rows * (rows - self.keep)) * stats.f.ppf(conf, self.keep, rows - self.keep)
        rest = self.var[self.keep:]
        th1, th2, th3 = rest.sum(), (rest ** 2).sum(), (rest ** 3).sum()
        h_0 = 1.0 - (2.0 * th1 * th3) / (3.0 * th2 * th2)
        self.spe_lim = th1 * (1.0 + th2 * h_0 * (h_0 - 1.0) / th1 ** 2
                              + stats.norm.ppf(conf) * np.sqrt(2.0 * th2 * h_0 ** 2) / th1) ** (1.0 / h_0)

    def flag(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with a boolean `flag` column."""
        z_data = (frame[COLUMNS].to_numpy(np.float64) - self.centre) / self.spread
        proj = z_data @ self.basis
        t2 = (proj ** 2 / self.var[:self.keep]).sum(axis=1)
        spe = ((z_data - proj @ self.basis.T) ** 2).sum(axis=1)
        return frame.assign(flag=(t2 > self.t2_lim) | (spe > self.spe_lim))

    def check(self, label: str, mine: float, theirs: float, tol: float) -> None:
        """Print and record one comparison."""
        ok = abs(float(mine) - float(theirs)) <= tol
        self.failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label:44s} recheck {float(mine):11.4f} | reported {float(theirs):11.4f}")


def alarm_samples(flagged: pd.DataFrame, keys: list[str], after: int) -> pd.Series:
    """Sample number of the alarm (third flag in a row, counted from sample > after) per run;
    NaN where there is none."""
    late = flagged[flagged["sample"] > after].sort_values(keys + ["sample"])
    streak = late.groupby(keys)["flag"].transform(lambda s: s.astype(int).rolling(IN_A_ROW).sum())
    hits = late[streak == IN_A_ROW]
    first = hits.groupby(keys)["sample"].min()
    return first.reindex(late.groupby(keys).size().index)


def jitter_sums(values: np.ndarray, window: int) -> np.ndarray:
    """Variance of the last `window` sample-to-sample changes from running sums; NaN until full."""
    out = np.full(len(values), np.nan)
    changes = np.diff(values)
    if len(changes) < window:
        return out
    run1 = np.concatenate(([0.0], np.cumsum(changes)))
    run2 = np.concatenate(([0.0], np.cumsum(changes * changes)))
    sum1, sum2 = run1[window:] - run1[:-window], run2[window:] - run2[:-window]
    out[window:] = (sum2 - sum1 * sum1 / window) / (window - 1)
    return out


def jitter_flag(frame: pd.DataFrame, column: str, window: int, limit: float) -> pd.DataFrame:
    """The frame, sorted, with `flag` = jitter above the limit (never flagged before the window fills)."""
    frame = frame.sort_values(["faultNumber", "simulationRun", "sample"]).reset_index(drop=True)
    values = frame[column].to_numpy(np.float64)
    jit = np.full(len(frame), np.nan)
    for rows in frame.groupby(["faultNumber", "simulationRun"]).indices.values():
        jit[rows] = jitter_sums(values[rows], window)
    return frame.assign(flag=np.where(np.isnan(jit), False, jit > limit))


def rates(flagged: pd.DataFrame, keys: list[str]) -> tuple[float, int, float]:
    """Flag % after 08:00 (mean over runs), runs with an alarm, median alarm delay in hours."""
    post = flagged[flagged["sample"] > ONSET]
    pct = 100 * post.groupby(keys)["flag"].mean().mean()
    hits = alarm_samples(flagged, keys, ONSET)
    delay = ((hits.dropna() - ONSET) * STEP_MIN / 60).median() if hits.notna().any() else float("nan")
    return pct, int(hits.notna().sum()), delay


def recheck_condenser(rc: "Recheck", normal_pca: pd.DataFrame) -> None:
    """Every number in out/condenser_check.json, the second way."""
    try:
        with open(OUT_DIR / "condenser_check.json", encoding="utf-8") as handle:
            cc = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"ERROR: run 05b_condenser_check.py first ({exc})")
        sys.exit(1)
    col = cc["signal"]["column"]
    train = read(PARQUET_DIR / "fault_free_training.parquet")
    normal = read(PARQUET_DIR / "fault_free_testing.parquet")
    faulty = read(PARQUET_DIR / "faulty_testing_runs1-20.parquet")
    held = read(PARQUET_DIR / "faulty15_testing_runs21-500.parquet")
    lo, hi = cc["holdout_runs"]
    rc.check("unseen runs: fault 15 only, runs 21-500", held["simulationRun"].nunique()
             if held["faultNumber"].eq(15).all() and held["simulationRun"].between(lo, hi).all() else -1,
             cc["holdout_run_count"], 0)
    pre = held[held["sample"] <= ONSET].merge(normal[normal["sample"] <= ONSET], on=["simulationRun", "sample"],
                                              suffixes=("", "_t"))
    same = np.all([np.array_equal(pre[c].to_numpy(), pre[c + "_t"].to_numpy()) for c in COLUMNS])
    rc.check("unseen runs identical to twin before 08:00", len(pre) / ONSET if same else -1,
             cc["holdout_twins_checked"], 0)
    twins_held = normal[normal["simulationRun"].between(lo, hi)]
    twins_dev = normal[normal["simulationRun"] <= cc["design_runs"][1]]
    for window, res in cc["windows"].items():
        win, tag = int(window), f"{res['window_hours']} h check"
        train_jit = np.concatenate([jitter_sums(g.sort_values("sample")[col].to_numpy(np.float64), win)
                                    for _r, g in train.groupby("simulationRun")])
        limit = float(np.quantile(train_jit[~np.isnan(train_jit)], cc["limit_percentile"] / 100))
        rc.check(f"{tag}: limit", limit, res["limit"], 1e-9)
        sets = [("unseen fault runs", held, res["holdout"]["fault"]),
                ("their twins", twins_held, res["holdout"]["twins"]),
                ("design fault runs 1-20", faulty[faulty["faultNumber"] == 15], res["design"]["fault"]),
                ("design twins 1-20", twins_dev, res["design"]["twins"]),
                ("500 normal runs", normal, res["normal"])]
        for label, frame, reported in sets:
            pct, alarmed, delay = rates(jitter_flag(frame, col, win, limit), ["simulationRun"])
            rc.check(f"{tag}: {label} flag %", pct, reported["flagged_pct_after_onset"], PCT_TOL)
            rc.check(f"{tag}: {label} runs alarmed", alarmed, reported["runs_alarmed"], 0)
            if reported["median_alarm_delay_h"] is not None:
                rc.check(f"{tag}: {label} median delay (h)", delay, reported["median_alarm_delay_h"], 0.051)
        others = jitter_flag(faulty[faulty["faultNumber"] != 15], col, win, limit)
        hits = alarm_samples(others, ["faultNumber", "simulationRun"], ONSET)
        per_fault = hits.notna().groupby(level=0).sum()
        mismatch = sum(int(per_fault.loc[int(f)]) != n for f, n in res["other_faults_runs_alarmed"].items())
        rc.check(f"{tag}: other faults, runs alarmed (mismatches)", mismatch, 0, 0)
        fires = sorted(int(f) for f, n in per_fault.items() if n >= cc["fires_on_share"] * 20)
        rc.check(f"{tag}: other faults it fires on (count)", len(fires) if fires == res["fires_on_faults"] else -1,
                 len(res["fires_on_faults"]), 0)
    pca_held = rc.flag(held)
    for label, frame, reported in [("unseen fault runs", pca_held, cc["pca_holdout"]["fault"]),
                                   ("their twins", normal_pca[normal_pca["simulationRun"].between(lo, hi)],
                                    cc["pca_holdout"]["twins"])]:
        pct, alarmed, _delay = rates(frame, ["simulationRun"])
        rc.check(f"PCA: {label} flag %", pct, reported["flagged_pct_after_onset"], PCT_TOL)
        rc.check(f"PCA: {label} runs alarmed", alarmed, reported["runs_alarmed"], 0)


def recheck_more(rc: "Recheck") -> None:
    """Every number in out/more_checks.json, the second way: pandas rolling windows (not running sums), PCA by SVD
    (not eigh), the CUSUM by itertools.accumulate (not a loop), limits by np.quantile, alarms by rolling windows."""
    import itertools  # pylint: disable=import-outside-toplevel
    try:
        with open(OUT_DIR / "more_checks.json", encoding="utf-8") as handle:
            mc = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"ERROR: run 05c_more_checks.py first ({exc})")
        sys.exit(1)
    train = read(PARQUET_DIR / "fault_free_training.parquet")
    normal = read(PARQUET_DIR / "fault_free_testing.parquet")
    faulty = read(PARQUET_DIR / "faulty_testing_runs1-20.parquet")
    held = read(PARQUET_DIR / "faulty15_testing_runs21-500.parquet")
    lo, hi = mc["holdout_runs"]

    def runs_of(frame):
        return [g.sort_values("sample") for _, g in frame.groupby(["faultNumber", "simulationRun"])]

    def logvar_pd(g, cols, w):
        ch = g[cols].astype(np.float64).diff()
        return np.log(ch.rolling(w, min_periods=w).var().clip(lower=1e-12)).to_numpy()

    def stats_multi(g, s):
        lv = logvar_pd(g, s["readings"], s["window_samples"])
        return ((lv - s["_mu"]) / s["_sd"]).sum(axis=1) / np.sqrt(len(s["readings"]))

    def stats_cusum(g, s):
        z = (g[s["reading"]].astype(np.float64).diff() ** 2 / s["normal_mean_change_sq"] - 1).to_numpy()[1:]
        acc = list(itertools.accumulate(z, lambda a, zi: max(0.0, a + zi - s["k"]), initial=0.0))
        return np.concatenate([[np.nan], acc[1:]])

    def stats_all52(g, s):
        f = logvar_pd(g, COLUMNS, s["window_samples"])
        zf = (f - s["_mu"]) / s["_sd"]
        t = zf @ s["_basis"]
        t2 = (t ** 2 / s["_var"]).sum(axis=1)
        spe = ((zf - t @ s["_basis"].T) ** 2).sum(axis=1)
        return np.maximum(t2 / s["t2_limit"], spe / s["spe_limit"])

    tr_runs = runs_of(train)
    sets = {"design_fault": faulty[faulty["faultNumber"] == 15], "design_twins": normal[normal["simulationRun"] <= 20],
            "holdout_fault": held, "holdout_twins": normal[normal["simulationRun"].between(lo, hi)], "normal": normal}
    for key, res in mc["checks"].items():
        s = dict(res["settings"])
        if key == "multi":
            lv = np.vstack([logvar_pd(g, s["readings"], s["window_samples"]) for g in tr_runs])
            s["_mu"], s["_sd"] = np.nanmean(lv, axis=0), np.nanstd(lv, axis=0)
            fn = stats_multi
        elif key == "cusum":
            ch2 = np.concatenate([g[s["reading"]].astype(np.float64).diff().to_numpy()[1:] ** 2 for g in tr_runs])
            rc.check("more: CUSUM normal mean change²", ch2.mean(), s["normal_mean_change_sq"], 1e-9 * ch2.mean())
            d15 = sets["design_fault"]
            exc = np.concatenate([(g[s["reading"]].astype(np.float64).diff().to_numpy()[1:] ** 2 / ch2.mean() - 1)[ONSET:]
                                  for g in runs_of(d15)])
            rc.check("more: CUSUM k (half the design excess)", 0.5 * exc.mean(), s["k"], 1e-9)
            fn = stats_cusum
        else:
            a = np.vstack([logvar_pd(g, COLUMNS, s["window_samples"]) for g in tr_runs])
            a = a[~np.isnan(a).any(axis=1)]
            mu, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
            z = (a - mu) / sd
            _u, sing, vt = np.linalg.svd(z, full_matrices=False)
            var = sing ** 2 / (z.shape[0] - 1)
            npc = int(np.argmax(var.cumsum() / var.sum() >= 0.90)) + 1
            rc.check("more: ALL52 components", npc, s["components"], 0)
            s["_mu"], s["_sd"], s["_basis"], s["_var"] = mu, sd, vt[:npc].T, var[:npc]
            t = z @ s["_basis"]
            rc.check("more: ALL52 T² limit", float(np.quantile((t ** 2 / s["_var"]).sum(axis=1), 0.99)), s["t2_limit"],
                     1e-6 * s["t2_limit"])
            rc.check("more: ALL52 SPE limit", float(np.quantile(((z - t @ s["_basis"].T) ** 2).sum(axis=1), 0.99)),
                     s["spe_limit"], 1e-6 * s["spe_limit"])
            fn = stats_all52
        tr_stats = np.concatenate([fn(g, s) for g in tr_runs])
        limit = float(np.quantile(tr_stats[~np.isnan(tr_stats)], mc["limit_percentile"] / 100))
        rc.check(f"more: {key} limit", limit, res["limit"], 1e-6 * max(1.0, abs(res["limit"])))

        def flagged(frame, fn=fn, s=s, limit=limit):
            parts = []
            for g in runs_of(frame):
                st = fn(g, s)
                parts.append(g[["faultNumber", "simulationRun", "sample"]].assign(flag=np.where(np.isnan(st), False, st > limit)))
            return pd.concat(parts, ignore_index=True)
        for label, frame in sets.items():
            pct, alarmed, delay = rates(flagged(frame), ["simulationRun"])
            rep = res[label]
            rc.check(f"more: {key} {label} flag %", pct, rep["flagged_pct_after_onset"], PCT_TOL)
            rc.check(f"more: {key} {label} runs alarmed", alarmed, rep["runs_alarmed"], 0)
            if rep["median_alarm_delay_h"] is not None:
                rc.check(f"more: {key} {label} median delay (h)", delay, rep["median_alarm_delay_h"], 0.051)
        hits = alarm_samples(flagged(faulty[faulty["faultNumber"] != 15]), ["faultNumber", "simulationRun"], ONSET)
        per_fault = hits.notna().groupby(level=0).sum()
        mismatch = sum(int(per_fault.loc[int(f)]) != n for f, n in res["other_faults_runs_alarmed"].items())
        rc.check(f"more: {key} other faults, runs alarmed (mismatches)", mismatch, 0, 0)


def std_ratio(fault_frame: pd.DataFrame, twin_frame: pd.DataFrame, col: str) -> float:
    """Median over runs of post-onset std (fault) / std (twin)."""
    f_sd = fault_frame[fault_frame["sample"] > ONSET].groupby("simulationRun")[col].std()
    t_sd = twin_frame[twin_frame["sample"] > ONSET].groupby("simulationRun")[col].std()
    return float((f_sd / t_sd.reindex(f_sd.index)).median())


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Recompute and compare."""
    rc = Recheck()
    mon, page = rc.mon, rc.page
    rc.check("components", rc.keep, mon["components"], 0)
    rc.check("T² limit", rc.t2_lim, mon["t2_limit"], 1e-6)
    rc.check("SPE limit", rc.spe_lim, mon["spe_limit"], 1e-6)

    normal = rc.flag(read(PARQUET_DIR / "fault_free_testing.parquet"))
    rc.check("normal-day flag % (48 h)", 100 * normal["flag"].mean(), mon["normal_flag_pct"]["any"], PCT_TOL)
    rc.check("normal flag % after 08:00", 100 * normal.loc[normal["sample"] > ONSET, "flag"].mean(),
             mon["normal_flag_pct_after_onset"], PCT_TOL)
    whole = alarm_samples(normal, ["simulationRun"], 0)
    rc.check("normal runs with an alarm (48 h)", whole.notna().sum(), mon["normal_runs_with_sustained_alarm"], 0)
    late = alarm_samples(normal, ["simulationRun"], ONSET)
    rc.check("twins 1-20 with an alarm after 08:00", late[late.index <= 20].notna().sum(),
             mon["twin_runs_1_20_with_alarm_after_onset"], 0)

    faulty = rc.flag(read(PARQUET_DIR / "faulty_testing_runs1-20.parquet"))
    twins = normal[normal["simulationRun"] <= 20]
    pre = faulty[faulty["sample"] <= ONSET].merge(twins[twins["sample"] <= ONSET], on=["simulationRun", "sample"],
                                                  suffixes=("", "_t"))
    same = np.all([np.array_equal(pre[c].to_numpy(), pre[c + "_t"].to_numpy()) for c in COLUMNS])
    rc.check("faulty runs identical to twin before 08:00", len(pre) / ONSET if same else -1,
             mon.get("twin_runs_checked", page["monitor"]["twin_runs_checked"]), 0)

    alarms = alarm_samples(faulty, ["faultNumber", "simulationRun"], ONSET)
    for fault in (14, 15):
        info = page["faults"][str(fault)]
        rows = faulty[faulty["faultNumber"] == fault]
        post = rows[rows["sample"] > ONSET]
        per_run = post.groupby("simulationRun")["flag"].mean() * 100
        rc.check(f"fault {fault} flag % after 08:00 (20 runs)", per_run.mean(), info["detection_rate_pct_all_runs"], PCT_TOL)
        hit = alarms.loc[fault]
        rc.check(f"fault {fault} runs that alarmed", hit.notna().sum(), info["runs_alarmed_of_20"], 0)
        if hit.notna().any():
            rc.check(f"fault {fault} median alarm delay (min)", ((hit.dropna() - ONSET) * STEP_MIN).median(),
                     info["median_alarm_delay_min"], 0)
        med = per_run.median()
        pick = sorted(per_run.index, key=lambda r: (abs(per_run[r] - med), r))[0]
        rc.check(f"fault {fault} representative run", pick, info["run"], 0)
        rc.check(f"fault {fault} run {pick} flag %", per_run[pick], info["run_detection_rate_pct"], PCT_TOL)
        twin_post = twins[(twins["simulationRun"] == pick) & (twins["sample"] > ONSET)]
        rc.check(f"fault {fault} run {pick} twin flag %", 100 * twin_post["flag"].mean(),
                 info["twin_detection_rate_pct_after_onset"], PCT_TOL)
        if info["alarm_sample"] is not None:
            rc.check(f"fault {fault} run {pick} alarm sample", hit.loc[pick], info["alarm_sample"], 0)
        for row in info["moved_most"][:3]:
            rc.check(f"fault {fault} spread ratio {row['tag']}", std_ratio(rows, twins, row["column"]),
                     row["spread_ratio"], RATIO_TOL)
        rest = max(std_ratio(rows, twins, c) for c in COLUMNS
                   if c not in [r["column"] for r in info["moved_most"][:3]])
        rc.check(f"fault {fault} largest other spread ratio", rest, info["moved_most"][3]["spread_ratio"], RATIO_TOL)
    f15 = page["faults"]["15"]
    rows15 = faulty[faulty["faultNumber"] == 15]
    rc.check("fault 15 spread ratio XMV(11)", std_ratio(rows15, twins, "xmv_11"),
             f15["spread_ratio_by_column"]["xmv_11"], RATIO_TOL)
    run6 = rows15[rows15["simulationRun"] == f15["run"]].sort_values("sample")
    twin6 = twins[twins["simulationRun"] == f15["run"]].sort_values("sample")
    hour = (run6["sample"].to_numpy() - 1) // 20
    f_sd = run6.groupby(hour)["xmeas_22"].std().to_numpy()
    t_sd = twin6.groupby(hour)["xmeas_22"].std().to_numpy()
    rc.check("fault 15 run hourly spread ratio XMEAS(22)", f_sd[8:].mean() / t_sd[8:].mean(),
             f15["spread_hourly"]["ratio_after_onset"], 0.0051)

    recheck_condenser(rc, normal)
    recheck_more(rc)

    if rc.failed:
        print("ERROR: recheck disagrees - do not publish these numbers")
        sys.exit(1)
    print("[DONE] independent recheck agrees on every quoted number")


if __name__ == "__main__":
    main()
