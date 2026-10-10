# PWW-DISCLOSURE: public
"""exp-006 load-chain cross-validation of the dense-spine (X-DSE) L5/S1
moment against the independently computed inverse-dynamics (X) L5/S1
moment (biomech_L5S1_Nm) on the full 6,007-pose cohort.

Fact-only. Success/failure is /pw-verify prerogative.

Recomputes T05 (load-chain endpoint agreement r) from the frozen
batch_v1 artefacts. Two per-pose scalar streams are merged on
file_stem, non-null rows retained, Pearson r reported with Fisher-z
95 percent CI, and per-domain r stratification recorded.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find(base_name: str, *rel_parts: str) -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.joinpath(*rel_parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(base_name)


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not -1 < r < 1:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    q = 1.959963984540054
    lo, hi = z - q * se, z + q * se
    return math.tanh(lo), math.tanh(hi)


def main() -> None:
    dse = pd.read_csv(_find("dense_spine.csv", "ace_pose", "_runs", "batch_v1", "analysis", "dense_spine.csv"))
    biomech = pd.read_csv(_find("summary_with_xw.csv", "ace_pose", "_runs", "batch_v1", "summary_with_xw.csv"))

    merged = dse.merge(
        biomech[["file_stem", "biomech_L5S1_Nm"]], on="file_stem", how="inner"
    )
    total = len(merged)
    merged = merged.dropna(subset=["spine_L5S1_moment_Nm", "biomech_L5S1_Nm"])
    kept = len(merged)

    x = merged["spine_L5S1_moment_Nm"].to_numpy(dtype=float)
    y = merged["biomech_L5S1_Nm"].to_numpy(dtype=float)
    r_overall = float(np.corrcoef(x, y)[0, 1])
    lo, hi = _fisher_ci(r_overall, kept)

    per_domain = {}
    for dom, sub in merged.groupby("domain"):
        xd = sub["spine_L5S1_moment_Nm"].to_numpy(dtype=float)
        yd = sub["biomech_L5S1_Nm"].to_numpy(dtype=float)
        r_d = float(np.corrcoef(xd, yd)[0, 1]) if len(sub) > 2 else float("nan")
        lo_d, hi_d = _fisher_ci(r_d, len(sub))
        per_domain[dom] = {
            "n": int(len(sub)),
            "pearson_r": r_d,
            "r_ci95_low": lo_d,
            "r_ci95_high": hi_d,
            "dse_mean_Nm": float(xd.mean()),
            "biomech_mean_Nm": float(yd.mean()),
        }

    metrics = {
        "n_total_rows": total,
        "n_kept_nonnull": kept,
        "n_dropped_null": total - kept,
        "pearson_r_overall": r_overall,
        "r_ci95_low": lo,
        "r_ci95_high": hi,
        "dse_L5S1_mean_Nm": float(x.mean()),
        "dse_L5S1_range_Nm": [float(x.min()), float(x.max())],
        "biomech_L5S1_mean_Nm": float(y.mean()),
        "biomech_L5S1_range_Nm": [float(y.min()), float(y.max())],
        "per_domain": per_domain,
        "inputs": {
            "dense_spine": "ace_pose/_runs/batch_v1/analysis/dense_spine.csv",
            "summary_xw": "ace_pose/_runs/batch_v1/summary_with_xw.csv",
        },
    }
    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
