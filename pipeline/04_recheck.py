"""
Independent re-check of EVERY measured number the page quotes (out/monitor.json,
out/measurements.csv, out/page_data.json).

Same definitions as 03_measure.py / 06_export_page.py, different code path: PCA by SVD of
the data matrix (not eigh of the covariance), limits re-coded from the formulas, alarms found
with rolling windows (not a loop), rates and ratios with pandas group operations, twins checked
with a merge. Must agree within tolerance, or exit 1. Run AFTER 06_export_page.py.
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

    if rc.failed:
        print("ERROR: recheck disagrees - do not publish these numbers")
        sys.exit(1)
    print("[DONE] independent recheck agrees on every quoted number")


if __name__ == "__main__":
    main()
