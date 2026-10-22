"""
Streaming reader for an .RData file that holds ONE data.frame of numeric columns.

Why it exists: TEP_Faulty_Testing.RData expands to ~4 GB of doubles. Loading it whole
(rdata / pyreadr) needs roughly twice that in RAM. This reader decompresses as a stream,
reads one column at a time, and keeps only the rows a caller selects, so peak memory is
a few columns, not the whole frame.

Scope (deliberately narrow): R's XDR serialization, format version 2 or 3, a top-level
pairlist holding one data.frame whose columns are REALSXP / INTSXP / LGLSXP vectors.
Anything else raises RFormatError with what was found, rather than guessing.
Checked against the general-purpose `rdata` package by pipeline/02_extract.py --check.
"""

import bz2
import gzip
import lzma
import struct
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

NILVALUE_SXP = 254
REFSXP = 255
SYMSXP = 1
LISTSXP = 2
CHARSXP = 9
LGLSXP = 10
INTSXP = 13
REALSXP = 14
STRSXP = 16
VECSXP = 19
ALTREP_SXP = 238
NUMERIC_DTYPES = {REALSXP: ">f8", INTSXP: ">i4", LGLSXP: ">i4"}
R_NA_INT = -(2**31)
CHUNK_ELEMS = 1 << 20


class RFormatError(Exception):
    """The file is not the narrow shape this reader supports."""


def open_decompressed(path: Path) -> BinaryIO:
    """Open an .RData file through the right decompressor, detected by magic bytes."""
    with open(path, "rb") as probe:
        magic = probe.read(6)
    if magic[:2] == b"\x1f\x8b":
        return gzip.open(path, "rb")
    if magic[:3] == b"BZh":
        return bz2.open(path, "rb")
    if magic[:6] == b"\xfd7zXZ\x00":
        return lzma.open(path, "rb")
    return open(path, "rb")


class _Reader:
    """Minimal XDR item reader with a reference table for symbols."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.refs: list[object] = []

    def read_exact(self, size: int) -> bytes:
        """Read exactly `size` bytes or raise."""
        buf = self.stream.read(size)
        if len(buf) != size:
            raise RFormatError(f"unexpected end of file (wanted {size} bytes, got {len(buf)})")
        return buf

    def read_int(self) -> int:
        """One big-endian signed 32-bit int."""
        return struct.unpack(">i", self.read_exact(4))[0]

    def read_length(self) -> int:
        """Vector length, including R's long-vector form."""
        length = self.read_int()
        if length == -1:
            upper, lower = self.read_int(), self.read_int()
            length = (upper << 32) + (lower & 0xFFFFFFFF)
        return length

    def read_header(self) -> None:
        """Consume 'RDX2/RDX3' + 'X' + version block."""
        sig = self.read_exact(5)
        if sig not in (b"RDX2\n", b"RDX3\n"):
            raise RFormatError(f"not an RData file (signature {sig!r})")
        fmt = self.read_exact(2)
        if fmt != b"X\n":
            raise RFormatError(f"only XDR format is supported (got {fmt!r})")
        version = self.read_int()
        self.read_int()  # writer R version
        self.read_int()  # minimal reader R version
        if version == 3:
            self.read_exact(self.read_int())  # native encoding name
        elif version != 2:
            raise RFormatError(f"serialization version {version} not supported")

    def read_item(self) -> object:
        """Read one small item fully (symbols, strings, small vectors, pairlists)."""
        flags = self.read_int()
        sxp = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if sxp == NILVALUE_SXP:
            return None
        if sxp == REFSXP:
            index = flags >> 8
            if index == 0:
                index = self.read_int()
            return self.refs[index - 1]
        if sxp == SYMSXP:
            name = self.read_item()
            self.refs.append(("sym", name))
            return ("sym", name)
        if sxp == CHARSXP:
            length = self.read_int()
            return None if length == -1 else self.read_exact(length).decode("utf-8", "replace")
        if sxp == LISTSXP:
            items = []
            while True:
                attr = self.read_item() if has_attr else None
                tag = self.read_item() if has_tag else None
                items.append((tag, self.read_item(), attr))
                flags = self.read_int()
                sxp = flags & 0xFF
                if sxp == NILVALUE_SXP:
                    return ("pairlist", items)
                if sxp != LISTSXP:
                    raise RFormatError(f"pairlist continued with type {sxp}, not supported")
                has_attr = bool(flags & (1 << 9))
                has_tag = bool(flags & (1 << 10))
        if sxp in NUMERIC_DTYPES:
            length = self.read_length()
            values = np.frombuffer(self.read_exact(length * 8 if sxp == REALSXP else length * 4),
                                   dtype=NUMERIC_DTYPES[sxp]).copy()
            attrs = self.read_item() if has_attr else None
            return ("vector", values, attrs)
        if sxp == STRSXP:
            length = self.read_length()
            values = [self.read_item() for _ in range(length)]
            attrs = self.read_item() if has_attr else None
            return ("strings", values, attrs)
        if sxp == ALTREP_SXP:
            raise RFormatError("ALTREP object found (file saved by R >= 3.5 with compact vectors)")
        raise RFormatError(f"item type {sxp} not supported by this narrow reader")

    def stream_column(self, keep: np.ndarray | None) -> np.ndarray:
        """Read one numeric column in chunks; return only rows where `keep` is True
        (all rows when `keep` is None)."""
        flags = self.read_int()
        sxp = flags & 0xFF
        if sxp not in NUMERIC_DTYPES:
            raise RFormatError(f"data.frame column of type {sxp}; only numeric columns supported")
        dtype = np.dtype(NUMERIC_DTYPES[sxp])
        length = self.read_length()
        parts = []
        done = 0
        while done < length:
            count = min(CHUNK_ELEMS, length - done)
            chunk = np.frombuffer(self.read_exact(count * dtype.itemsize), dtype=dtype)
            if keep is not None:
                chunk = chunk[keep[done:done + count]]
            parts.append(chunk.astype(np.float64 if sxp == REALSXP else np.int64))
            done += count
        if flags & (1 << 9):
            self.read_item()  # per-column attributes (none expected), consumed and dropped
        column = np.concatenate(parts) if parts else np.empty(0)
        if sxp != REALSXP:
            column = np.where(column == R_NA_INT, np.nan, column)
        return column


def read_dataframe(path: Path, row_filter: Callable[[dict[int, np.ndarray]], np.ndarray] | None,
                   filter_after: int) -> tuple[str, dict[str, np.ndarray]]:
    """Stream one data.frame out of `path`.

    `filter_after`: index of the last column the filter needs (e.g. 1 for simulationRun).
    Columns 0..filter_after are read whole; `row_filter` gets them and returns the mask.
    Returns (object name, {column name: values}).
    """
    with open_decompressed(path) as stream:
        reader = _Reader(stream)
        reader.read_header()
        flags = reader.read_int()
        if flags & 0xFF != LISTSXP or not flags & (1 << 10):
            raise RFormatError("top level is not a tagged pairlist (not a save() file?)")
        if flags & (1 << 9):
            reader.read_item()
        tag = reader.read_item()
        obj_name = tag[1] if isinstance(tag, tuple) else str(tag)

        flags = reader.read_int()
        if flags & 0xFF != VECSXP:
            raise RFormatError(f"object '{obj_name}' is type {flags & 0xFF}, not a data.frame list")
        n_cols = reader.read_length()
        columns: list[np.ndarray] = []
        keep = None
        for index in range(n_cols):
            if row_filter is not None and index <= filter_after:
                columns.append(reader.stream_column(None))
                if index == filter_after:
                    keep = row_filter(dict(enumerate(columns)))
                    columns = [col[keep] for col in columns]
            else:
                columns.append(reader.stream_column(keep))
        attrs = reader.read_item() if flags & (1 << 9) else None
        if reader.read_int() & 0xFF != NILVALUE_SXP:
            raise RFormatError("more than one object in the file; this reader expects exactly one")

    names = None
    if attrs and attrs[0] == "pairlist":
        for tag_item, value, _ in attrs[1]:
            if isinstance(tag_item, tuple) and tag_item[1] == "names":
                names = value[1]
    if names is None or len(names) != n_cols:
        raise RFormatError("could not read the data.frame column names")
    return obj_name, dict(zip(names, columns))
