# PWW-DISCLOSURE: public
"""Compare two result folders file by file (CSV, JSON, NPZ) with a numeric tolerance.

Usage: python topic/E/src/compare_results.py REFERENCE_DIR NEW_DIR [--rtol 1e-9]
Prints one line per file (identical / within tolerance / DIFFERENT / missing)
and exits 1 if any file differs or is missing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cmp_values(a, b, rtol):
    """Return max relative difference, or None if structurally different."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return None
        ds = [cmp_values(a[k], b[k], rtol) for k in a]
        return None if None in ds else max(ds, default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        ds = [cmp_values(x, y, rtol) for x, y in zip(a, b)]
        return None if None in ds else max(ds, default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == b or (np.isnan(a) and np.isnan(b)):
            return 0.0
        return abs(a - b) / max(abs(a), abs(b), 1e-300)
    return 0.0 if a == b else None


def cmp_frames(a: pd.DataFrame, b: pd.DataFrame):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return None
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            same_nan = np.isnan(x) == np.isnan(y)
            if not same_nan.all():
                return None
            m = ~np.isnan(x)
            d = np.abs(x[m] - y[m]) / np.maximum(np.maximum(np.abs(x[m]), np.abs(y[m])), 1e-300)
            worst = max(worst, float(d.max(initial=0.0)))
        elif not x.astype(str).equals(y.astype(str)):
            return None
    return worst


def compare(ref: Path, new: Path, rtol: float) -> int:
    bad = 0
    for p in sorted(ref.rglob("*")):
        if not p.is_file() or p.suffix not in {".csv", ".json", ".npz"}:
            continue
        q = new / p.relative_to(ref)
        rel = p.relative_to(ref)
        if not q.is_file():
            print(f"missing    {rel}"); bad += 1; continue
        if p.suffix == ".csv":
            d = cmp_frames(pd.read_csv(p), pd.read_csv(q))
        elif p.suffix == ".json":
            d = cmp_values(json.loads(p.read_text()), json.loads(q.read_text()), rtol)
        else:
            a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
            if set(a.files) != set(b.files):
                d = None
            else:
                ds = []
                for k in a.files:
                    x, y = a[k], b[k]
                    if x.shape != y.shape:
                        ds.append(None)
                    elif x.dtype.kind in "fiu":
                        diff = np.abs(x.astype(float) - y.astype(float))
                        ds.append(float(np.nanmax(diff / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-300),
                                                  initial=0.0)))
                    else:
                        ds.append(0.0 if np.array_equal(x, y) else None)
                d = None if None in ds else max(ds, default=0.0)
        if d is None or d > rtol:
            print(f"DIFFERENT  {rel}" + ("" if d is None else f"  (max rel diff {d:.2e})")); bad += 1
        else:
            print(f"{'identical' if d == 0 else 'within tol':10s} {rel}" + ("" if d == 0 else f"  ({d:.1e})"))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-9)
    a = ap.parse_args()
    n = compare(a.ref, a.new, a.rtol)
    print(f"\n{n} file(s) differ or are missing")
    sys.exit(1 if n else 0)
