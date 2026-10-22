"""
Fit a standard PCA process monitor on normal running, then measure what it sees.

Definitions (stated once, used everywhere):
- Model: PCA on the 52 standardised variables of TEP_FaultFree_Training (500 runs x 500
  samples). Components kept: the fewest explaining >= CPV_TARGET of variance.
- Statistics: Hotelling T² (inside the model) and SPE / Q (outside it), 99% limits.
  T² limit: a(n-1)(n+1)/(n(n-a)) * F(a, n-a). SPE limit: Jackson & Mudholkar (1979).
- A sample is FLAGGED when T² or SPE exceeds its limit. An ALARM is 3 flagged samples in a
  row, and it sounds at the THIRD (that is when an operator could first see it).
- False-alarm rate (FAR): share of flagged samples in TEP_FaultFree_Testing (500 runs x 960),
  over the whole run and over the same hours as the faults (sample > 160).
- Detection rate (FDR): share of flagged samples after the fault starts (sample > 160,
  i.e. after hour 8), runs 1-20 of each fault.
- Alarm delay: minutes from fault start to the alarm (the third flag in a row).

Writes out/measurements.csv and out/monitor.json. Re-check: pipeline/04_recheck.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tags import FAULTS, catalogue

LAB_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = LAB_DIR / "data" / "parquet"
OUT_DIR = LAB_DIR / "out"
VARIABLES = [row["column"] for row in catalogue()]
CPV_TARGET = 0.90
CONFIDENCE = 0.99
ONSET_SAMPLE = 160          # testing runs: fault introduced after sample 160 (hour 8)
SAMPLE_MINUTES = 3
CONSECUTIVE = 3


def load(name: str) -> pd.DataFrame:
    """Load one extracted Parquet file or exit with what to run."""
    path = PARQUET_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found. Run pipeline/02_extract.py first.")
        sys.exit(1)
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {path} - {exc}")
        sys.exit(1)
    missing = [v for v in VARIABLES if v not in frame.columns]
    if missing:
        print(f"ERROR: {path} lacks columns {missing[:5]}")
        sys.exit(1)
    return frame


class PcaMonitor:
    """PCA monitor: standardise, project, T² and SPE with 99% limits."""

    def __init__(self, train: np.ndarray, cpv_target: float, confidence: float) -> None:
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0, ddof=1)
        if np.any(self.std == 0):
            raise ValueError("a variable is constant in the training data")
        scaled = (train - self.mean) / self.std
        n_samples = scaled.shape[0]
        eigvals, eigvecs = np.linalg.eigh(np.cov(scaled, rowvar=False))
        order = np.argsort(eigvals)[::-1]
        self.eigvals, self.eigvecs = eigvals[order], eigvecs[:, order]
        cpv = np.cumsum(self.eigvals) / self.eigvals.sum()
        self.n_comp = int(np.searchsorted(cpv, cpv_target) + 1)
        self.cpv = float(cpv[self.n_comp - 1])
        self.loadings = self.eigvecs[:, :self.n_comp]
        comp = self.n_comp
        self.t2_limit = float(comp * (n_samples - 1) * (n_samples + 1)
                              / (n_samples * (n_samples - comp))
                              * stats.f.ppf(confidence, comp, n_samples - comp))
        resid = self.eigvals[comp:]
        theta1, theta2, theta3 = (np.sum(resid ** k) for k in (1, 2, 3))
        h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 ** 2)
        z_alpha = stats.norm.ppf(confidence)
        self.spe_limit = float(theta1 * (z_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1
                                         + 1 + theta2 * h0 * (h0 - 1) / theta1 ** 2) ** (1 / h0))

    def score(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (T², SPE) for each row."""
        scaled = (data - self.mean) / self.std
        scores = scaled @ self.loadings
        t2 = np.sum(scores ** 2 / self.eigvals[:self.n_comp], axis=1)
        resid = scaled - scores @ self.loadings.T
        return t2, np.sum(resid ** 2, axis=1)

    def flags(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Boolean flags per row for T², SPE and either."""
        t2, spe = self.score(data)
        t2_flag, spe_flag = t2 > self.t2_limit, spe > self.spe_limit
        return {"t2": t2_flag, "spe": spe_flag, "any": t2_flag | spe_flag}


def alarm_index(flagged: np.ndarray, start: int, run_length: int) -> int | None:
    """Index of the sample that COMPLETES the first run of `run_length` consecutive flags
    at or after `start` (the moment the alarm sounds), or None."""
    streak = 0
    for index in range(start, len(flagged)):
        streak = streak + 1 if flagged[index] else 0
        if streak == run_length:
            return index
    return None


def measure_faults(monitor: PcaMonitor, faulty: pd.DataFrame) -> pd.DataFrame:
    """Per-fault detection rate, pre-fault false alarms and detection delay."""
    rows = []
    for fault_id, group in faulty.groupby("faultNumber"):
        post, pre, delays, detected = {"t2": [], "spe": [], "any": []}, [], [], 0
        for _run, run in group.groupby("simulationRun"):
            run = run.sort_values("sample")
            flagged = monitor.flags(run[VARIABLES].to_numpy(dtype=np.float64))
            after = run["sample"].to_numpy() > ONSET_SAMPLE
            for key in post:
                post[key].append(flagged[key][after].mean())
            pre.append(flagged["any"][~after].mean())
            onset_pos = int(np.argmax(after))
            hit = alarm_index(flagged["any"], onset_pos, CONSECUTIVE)
            if hit is not None:
                detected += 1
                # sample onset_pos is 3 min after the fault starts, so index k is (k-onset_pos+1)*3 min
                delays.append((hit - onset_pos + 1) * SAMPLE_MINUTES)
        kind, what = FAULTS[int(fault_id)]
        rows.append({
            "fault": int(fault_id), "type": kind, "disturbed": what,
            "detection_rate_any_pct": round(100 * float(np.mean(post["any"])), 2),
            "detection_rate_t2_pct": round(100 * float(np.mean(post["t2"])), 2),
            "detection_rate_spe_pct": round(100 * float(np.mean(post["spe"])), 2),
            "pre_fault_flag_pct": round(100 * float(np.mean(pre)), 2),
            "runs_alarmed_of_20": detected,
            "median_alarm_delay_min": float(np.median(delays)) if delays else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Fit, measure, write."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load("fault_free_training.parquet")[VARIABLES].to_numpy(dtype=np.float64)
    try:
        monitor = PcaMonitor(train, CPV_TARGET, CONFIDENCE)
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: PCA fit failed - {exc}")
        sys.exit(1)
    print(f"[STEP] PCA: {monitor.n_comp} components explain {100 * monitor.cpv:.1f}% "
          f"| T² limit {monitor.t2_limit:.2f} | SPE limit {monitor.spe_limit:.2f}")

    normal = load("fault_free_testing.parquet")
    normal_flags = monitor.flags(normal[VARIABLES].to_numpy(dtype=np.float64))
    far = {key: round(100 * float(val.mean()), 2) for key, val in normal_flags.items()}
    after_onset = normal["sample"].to_numpy() > ONSET_SAMPLE
    far_after = round(100 * float(normal_flags["any"][after_onset].mean()), 2)
    sustained_runs, sustained_after, twins_after = 0, 0, 0
    for run_no, run in normal.assign(flag=normal_flags["any"]).groupby("simulationRun"):
        run = run.sort_values("sample")
        flags = run["flag"].to_numpy()
        onset_pos = int(np.argmax(run["sample"].to_numpy() > ONSET_SAMPLE))
        sustained_runs += alarm_index(flags, 0, CONSECUTIVE) is not None
        late = alarm_index(flags, onset_pos, CONSECUTIVE) is not None
        sustained_after += late
        twins_after += late and run_no <= 20
    print(f"[STEP] Normal running ({len(normal):,} samples): flagged {far['any']}% "
          f"(T² {far['t2']}%, SPE {far['spe']}%), after 08:00 {far_after}% | runs with an alarm: "
          f"{sustained_runs} of {normal['simulationRun'].nunique()} (after 08:00: {sustained_after}; "
          f"twins 1-20 after 08:00: {twins_after})")

    table = measure_faults(monitor, load("faulty_testing_runs1-20.parquet"))
    try:
        table.to_csv(OUT_DIR / "measurements.csv", index=False, encoding="utf-8")
        with open(OUT_DIR / "monitor.json", "w", encoding="utf-8") as handle:
            json.dump({
                "dataset": "Rieth et al. 2017, doi:10.7910/DVN/6C3JR1",
                "fit_on": "TEP_FaultFree_Training, 500 runs x 500 samples",
                "variables": len(VARIABLES), "cpv_target": CPV_TARGET,
                "components": monitor.n_comp, "cpv": round(monitor.cpv, 4),
                "confidence": CONFIDENCE, "t2_limit": monitor.t2_limit,
                "spe_limit": monitor.spe_limit,
                "normal_flag_pct": far, "normal_flag_pct_after_onset": far_after,
                "normal_runs_with_sustained_alarm": sustained_runs,
                "normal_runs_with_alarm_after_onset": sustained_after,
                "twin_runs_1_20_with_alarm_after_onset": twins_after,
                "normal_runs": int(normal["simulationRun"].nunique()),
                "onset_sample": ONSET_SAMPLE, "sample_minutes": SAMPLE_MINUTES,
                "consecutive_for_alarm": CONSECUTIVE,
            }, handle, indent=2)
    except OSError as exc:
        print(f"ERROR: cannot write results to {OUT_DIR} - {exc}")
        sys.exit(1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.drop(columns=["disturbed"]).to_string(index=False))
    print(f"[DONE] {OUT_DIR / 'measurements.csv'}")


if __name__ == "__main__":
    main()
