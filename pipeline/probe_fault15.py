"""
Probe: does the sticking condenser valve (fault 15) leave ANY trace in its own loop?

Checks, per run, before vs after the fault starts:
- the condenser cooling-water valve signal XMV(11) and condenser outlet temperature XMEAS(22):
  standard deviation and lag-1 autocorrelation (a stuck valve often makes a loop hunt);
- the same for fault 14 (sticking REACTOR cooling-water valve), which the monitor does catch,
  as a positive control, and fault-free runs as a negative control.
Also checks whether the first 160 samples are identical across faults for the same run number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = LAB_DIR / "data" / "parquet"
ONSET = 160
SIGNALS = {
    15: ["xmv_11", "xmeas_22", "xmeas_11", "xmeas_13"],
    14: ["xmv_10", "xmeas_21", "xmeas_9"],
}


def load(name: str) -> pd.DataFrame:
    """Load a Parquet file or exit."""
    try:
        return pd.read_parquet(PARQUET_DIR / name)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {PARQUET_DIR / name} - {exc}")
        sys.exit(1)


def lag1(values: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    centred = values - values.mean()
    return float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))


def summarise(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    """Print median-over-runs std and lag-1 autocorrelation, before vs after onset."""
    print(f"\n{label}")
    for column in columns:
        ratios, before_ac, after_ac = [], [], []
        for _run, run in frame.groupby("simulationRun"):
            vals = run.sort_values("sample")[column].to_numpy(dtype=np.float64)
            pre, post = vals[:ONSET], vals[ONSET:]
            ratios.append(post.std(ddof=1) / pre.std(ddof=1))
            before_ac.append(lag1(pre))
            after_ac.append(lag1(post))
        print(f"  {column:9s} std after/before (median of runs) {np.median(ratios):5.2f} "
              f"| lag-1 autocorr before {np.median(before_ac):5.2f} after {np.median(after_ac):5.2f}")


def main() -> None:
    """Run the probes."""
    faulty = load("faulty_testing_runs1-20.parquet")
    normal = load("fault_free_testing.parquet")
    normal = normal[normal["simulationRun"] <= 20]

    first = faulty[faulty["sample"] <= ONSET]
    run1 = first[first["simulationRun"] == 1]
    base = run1[run1["faultNumber"] == 1].drop(columns="faultNumber").reset_index(drop=True)
    same = [int(f) for f, grp in run1.groupby("faultNumber")
            if grp.drop(columns="faultNumber").reset_index(drop=True).equals(base)]
    print(f"Run 1, samples 1-160: identical to fault 1's for faults {same}")
    norm1 = normal[(normal["simulationRun"] == 1) & (normal["sample"] <= ONSET)]
    print(f"Fault-free run 1 identical to faulty run 1 pre-onset: "
          f"{norm1.drop(columns='faultNumber').reset_index(drop=True).equals(base)}")

    for fault, columns in SIGNALS.items():
        summarise(faulty[faulty["faultNumber"] == fault], columns, f"FAULT {fault}")
        summarise(normal, columns, f"NORMAL (control) - same columns as fault {fault}")


if __name__ == "__main__":
    main()
