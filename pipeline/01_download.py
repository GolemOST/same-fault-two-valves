"""
Download the public Tennessee Eastman dataset (Rieth et al. 2017, Harvard Dataverse).

Checks the licence text in the dataset metadata, streams each file to data/raw/,
and verifies size + checksum before keeping it. Re-running skips files already verified.

Usage:
    python pipeline/01_download.py fault-free
    python pipeline/01_download.py faulty-testing
    python pipeline/01_download.py faulty-training
"""

import hashlib
import json
import sys
from pathlib import Path

import requests

DOI = "doi:10.7910/DVN/6C3JR1"
API_BASE = "https://dataverse.harvard.edu/api"
LAB_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = LAB_DIR / "data" / "raw"
GROUPS = {
    "fault-free": ["TEP_FaultFree_Training.RData", "TEP_FaultFree_Testing.RData"],
    "faulty-testing": ["TEP_Faulty_Testing.RData"],
    "faulty-training": ["TEP_Faulty_Training.RData"],
}
CHUNK = 1024 * 1024
# Dataverse returns 403 to the default python-requests User-Agent.
HEADERS = {"User-Agent": "benchmark-plant-lab/0.1 (one-off research download of a public dataset)"}


def fetch_metadata() -> dict:
    """Return the dataset's latest-version metadata from the Dataverse API."""
    url = f"{API_BASE}/datasets/:persistentId/"
    try:
        response = requests.get(url, params={"persistentId": DOI}, headers=HEADERS, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: could not read dataset metadata from {url} ({DOI}) - {exc}")
        sys.exit(1)
    try:
        return response.json()["data"]["latestVersion"]
    except (ValueError, KeyError) as exc:
        print(f"ERROR: metadata from {url} is not the expected JSON shape - {exc}")
        sys.exit(1)


def licence_text(version: dict) -> str:
    """Pull whatever licence / terms field the API returned, as one string."""
    parts = []
    licence = version.get("license")
    if isinstance(licence, dict):
        parts.append(f"{licence.get('name', '')} {licence.get('uri', '')}".strip())
    elif licence:
        parts.append(str(licence))
    for key in ("termsOfUse", "termsOfAccess"):
        if version.get(key):
            parts.append(str(version[key]))
    return " | ".join(parts) or "(no licence field returned)"


def file_md5(path: Path) -> str:
    """MD5 of a local file, streamed."""
    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
    except OSError as exc:
        print(f"ERROR: cannot read {path} to checksum it - {exc}")
        sys.exit(1)
    return digest.hexdigest()


def expected_checksum(data_file: dict) -> tuple[str, str]:
    """Return (algorithm, value) from the Dataverse file record, or ('', '')."""
    checksum = data_file.get("checksum") or {}
    if checksum.get("type") and checksum.get("value"):
        return checksum["type"].upper(), checksum["value"]
    if data_file.get("md5"):
        return "MD5", data_file["md5"]
    return "", ""


def verify(path: Path, data_file: dict) -> bool:
    """True when size and (if published) MD5 match the Dataverse record."""
    size = path.stat().st_size
    if size != data_file["filesize"]:
        print(f"  size mismatch: local {size:,} vs published {data_file['filesize']:,}")
        return False
    algo, value = expected_checksum(data_file)
    if algo == "MD5":
        local = file_md5(path)
        if local != value:
            print(f"  MD5 mismatch: local {local} vs published {value}")
            return False
        print(f"  MD5 OK {local}")
    elif algo:
        print(f"  published checksum is {algo}, not checked (size matched)")
    return True


def download(data_file: dict, target: Path) -> None:
    """Stream one file to target via a .part file; rename only after it verifies."""
    url = f"{API_BASE}/access/datafile/{data_file['id']}"
    part = target.with_suffix(target.suffix + ".part")
    print(f"[STEP] Downloading {target.name} ({data_file['filesize'] / 1e6:,.1f} MB) from {url}")
    try:
        with requests.get(url, stream=True, headers=HEADERS, timeout=120) as response:
            response.raise_for_status()
            done = 0
            with open(part, "wb") as handle:
                for block in response.iter_content(chunk_size=CHUNK):
                    handle.write(block)
                    done += len(block)
                    if done % (100 * CHUNK) < CHUNK:
                        print(f"  {done / 1e6:,.0f} MB")
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: download failed for {url} - {exc}. Partial file kept at {part}")
        sys.exit(1)
    except OSError as exc:
        print(f"ERROR: cannot write {part} - {exc}. Check free disk space.")
        sys.exit(1)
    if not verify(part, data_file):
        print(f"ERROR: {part} failed verification; not renamed. Delete it and re-run.")
        sys.exit(1)
    try:
        part.replace(target)
    except OSError as exc:
        print(f"ERROR: cannot rename {part} -> {target} - {exc}")
        sys.exit(1)
    print(f"  saved {target}")


def main() -> None:
    """Fetch metadata, show the licence, download the requested group."""
    if len(sys.argv) != 2 or sys.argv[1] not in GROUPS:
        print(f"Usage: python {Path(__file__).name} [{'|'.join(GROUPS)}]")
        sys.exit(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    version = fetch_metadata()
    terms = licence_text(version)
    print(f"[STEP] Licence / terms as published today: {terms}")
    if "public domain" not in terms.lower() and "cc0" not in terms.lower():
        print("ERROR: licence text no longer says public domain / CC0. Stop and re-check by hand.")
        sys.exit(1)

    try:
        with open(RAW_DIR / "_dataset_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(version, handle, indent=2)
    except OSError as exc:
        print(f"ERROR: cannot write metadata record to {RAW_DIR} - {exc}")
        sys.exit(1)

    records = {f["dataFile"]["filename"]: f["dataFile"] for f in version.get("files", [])}
    for name in GROUPS[sys.argv[1]]:
        if name not in records:
            print(f"ERROR: {name} is not in the dataset file list: {sorted(records)}")
            sys.exit(1)
        target = RAW_DIR / name
        if target.exists():
            print(f"[STEP] {name} already present, re-verifying")
            if verify(target, records[name]):
                continue
            print(f"ERROR: existing {target} failed verification. Delete it and re-run.")
            sys.exit(1)
        download(records[name], target)
    print("[DONE] all requested files present and verified")


if __name__ == "__main__":
    main()
