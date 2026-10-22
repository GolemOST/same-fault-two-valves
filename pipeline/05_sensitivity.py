"""
Does the headline depend on how the monitor was tuned? Re-run faults 14 and 15 and the
normal day with the variance target at 80%, 90% (the default) and 95%, and at 99.9% limits.
Writes out/sensitivity.csv.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
measure = importlib.import_module("03_measure")

SETTINGS = [(0.80, 0.99), (0.90, 0.99), (0.95, 0.99), (0.90, 0.999)]
FAULTS = (14, 15)


def main() -> None:
    """Sweep the settings and print one row each."""
    train = measure.load("fault_free_training.parquet")[measure.VARIABLES].to_numpy(np.float64)
    normal = measure.load("fault_free_testing.parquet")[measure.VARIABLES].to_numpy(np.float64)
    faulty = measure.load("faulty_testing_runs1-20.parquet")
    post = faulty[(faulty["sample"] > measure.ONSET_SAMPLE) & faulty["faultNumber"].isin(FAULTS)]
    rows = []
    for cpv, conf in SETTINGS:
        try:
            monitor = measure.PcaMonitor(train, cpv, conf)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"ERROR: PCA fit failed at cpv={cpv}, conf={conf} - {exc}")
            sys.exit(1)
        row = {"variance_target": cpv, "confidence": conf, "components": monitor.n_comp,
               "normal_day_flag_pct": round(100 * monitor.flags(normal)["any"].mean(), 2)}
        for fault in FAULTS:
            data = post[post["faultNumber"] == fault][measure.VARIABLES].to_numpy(np.float64)
            row[f"fault_{fault}_flag_pct"] = round(100 * monitor.flags(data)["any"].mean(), 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    out = measure.OUT_DIR / "sensitivity.csv"
    try:
        table.to_csv(out, index=False, encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write {out} - {exc}")
        sys.exit(1)
    print(table.to_string(index=False))
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
