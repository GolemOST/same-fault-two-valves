"""
Turn the Rieth 2017 .RData files into small Parquet files the analysis can load fast.

    python pipeline/02_extract.py --check   # prove rstream == rdata on a real file, then stop
    python pipeline/02_extract.py           # extract all three files to data/parquet/

Kept: every fault-free run (training + testing), and runs 1..RUNS_PER_FAULT of each fault in
Faulty_Testing. Process variables stored as float32 (source values carry ~5 significant
digits); ids as int16.
"""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from rstream import RFormatError, read_dataframe

LAB_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = LAB_DIR / "data" / "raw"
PARQUET_DIR = LAB_DIR / "data" / "parquet"
RUNS_PER_FAULT = 20
ID_COLUMNS = ("faultNumber", "simulationRun", "sample")
JOBS = [
    ("TEP_FaultFree_Training.RData", "fault_free_training.parquet", None),
    ("TEP_FaultFree_Testing.RData", "fault_free_testing.parquet", None),
    ("TEP_Faulty_Testing.RData", "faulty_testing_runs1-20.parquet", RUNS_PER_FAULT),
]


def require(path: Path) -> None:
    """Exit with a clear message if an input file is missing."""
    if not path.exists():
        print(f"ERROR: {path} not found. Run pipeline/01_download.py first.")
        sys.exit(1)


def stream(path: Path, max_run: int | None) -> dict[str, np.ndarray]:
    """Read one file with the streaming reader, optionally keeping runs <= max_run."""
    row_filter = None
    if max_run is not None:
        def row_filter(cols: dict[int, np.ndarray]) -> np.ndarray:
            return cols[1] <= max_run  # column 1 = simulationRun
    try:
        _name, columns = read_dataframe(path, row_filter, filter_after=1)
    except (RFormatError, OSError, EOFError) as exc:
        print(f"ERROR: streaming read of {path} failed - {exc}")
        sys.exit(1)
    if list(columns)[:3] != list(ID_COLUMNS):
        print(f"ERROR: {path} first columns are {list(columns)[:3]}, expected {ID_COLUMNS}")
        sys.exit(1)
    return columns


def check() -> None:
    """Compare rstream with the general-purpose rdata package on both fault-free files (the
    837 MB faulty file is too big to load whole with rdata on this machine)."""
    try:
        import rdata  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("ERROR: the check needs the `rdata` package: python -m pip install rdata")
        sys.exit(1)
    for raw_name, _out, _runs in JOBS[:2]:
        check_one(rdata, RAW_DIR / raw_name)


def check_one(rdata, path: Path) -> None:  # noqa: ANN001  (module object)
    """rstream vs rdata on one file, all rows and a run filter."""
    require(path)
    print(f"[STEP] rdata reference read of {path.name}")
    try:
        frame = next(iter(rdata.read_rda(path).values()))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: rdata could not read {path} - {exc}")
        sys.exit(1)
    for max_run in (None, 7):
        label = "all rows" if max_run is None else f"runs <= {max_run}"
        mine = stream(path, max_run)
        ref = frame if max_run is None else frame[frame["simulationRun"] <= max_run]
        if list(mine) != list(ref.columns):
            print(f"FAIL ({label}): column names differ")
            sys.exit(1)
        bad = [name for name in ref.columns
               if not np.array_equal(mine[name], ref[name].to_numpy(dtype=np.float64), equal_nan=True)]
        if bad or len(mine["sample"]) != len(ref):
            print(f"FAIL ({label}): {len(bad)} columns differ: {bad[:5]} rows {len(mine['sample'])} vs {len(ref)}")
            sys.exit(1)
        print(f"  PASS ({label}): {len(ref):,} rows x {len(ref.columns)} columns identical, bit for bit")


def to_frame(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """Columns -> compact DataFrame (int16 ids, float32 process variables)."""
    data = {}
    for name, values in columns.items():
        if name in ID_COLUMNS:
            if np.isnan(values).any():
                print(f"ERROR: id column {name} has missing values")
                sys.exit(1)
            data[name] = values.astype(np.int16)
        else:
            data[name] = values.astype(np.float32)
    return pd.DataFrame(data)


def extract() -> None:
    """Extract every job to Parquet and print what was kept."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for raw_name, out_name, max_run in JOBS:
        source = RAW_DIR / raw_name
        require(source)
        started = time.time()
        tracemalloc.start()
        frame = to_frame(stream(source, max_run))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        target = PARQUET_DIR / out_name
        try:
            frame.to_parquet(target, index=False)
        except (OSError, ValueError) as exc:
            print(f"ERROR: cannot write {target} - {exc}")
            sys.exit(1)
        faults = sorted(frame["faultNumber"].unique().tolist())
        runs = frame.groupby("faultNumber")["simulationRun"].nunique()
        samples = frame.groupby(["faultNumber", "simulationRun"]).size()
        print(f"[DONE] {out_name}: {len(frame):,} rows | faults {faults[0]}..{faults[-1]} ({len(faults)}) "
              f"| runs/fault {runs.min()}-{runs.max()} | samples/run {samples.min()}-{samples.max()} "
              f"| peak traced RAM {peak / 1e9:.2f} GB | {time.time() - started:.0f} s")


def main() -> None:
    """Entry point."""
    if "--check" in sys.argv:
        check()
    else:
        extract()


if __name__ == "__main__":
    main()
