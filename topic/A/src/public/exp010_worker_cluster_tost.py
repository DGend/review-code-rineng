# PWW-DISCLOSURE: public
"""exp-010: worker-clustered TOST equivalence test, manufacturing vs
eldercare RULA grand score, per charter E-CD3 / R-CD2 / R-CD3.

The §4.3 manuscript claim ("eldercare caregiving tasks are
ergonomically equivalent to industrial preparatory work") was derived
from a Kruskal-Wallis significance test, which shows a DIFFERENCE
exists, not that the domains are equivalent (R-CD2/R-CD3 reviewer
objection). This script runs the pre-registered TOST instead
(margin +/- 0.5 RULA action-level unit, E-CD3) and reports both a
naive (pose-level) bootstrap CI and a worker-clustered bootstrap CI
(R-CD3), because R-CD3 requires the delta between the two to be
published.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path as _Path; _ROOT = _Path(__file__).resolve().parents[4]  # project root
CSV_PATH = str(_ROOT) + "/ace_pose/_runs/batch_v1/summary_with_mc.csv"
MARGIN = 0.5  # E-CD3 primary TOST margin, RULA action-level units
N_BOOT = 1000
SEED = 20260821


def domain_of(row) -> str:
    f = str(row.get("factory_id", ""))
    d = str(row.get("domain", ""))
    return "eldercare" if (f == "F6" or d == "eldercare") else "manufacturing"


def tost(mean_diff: float, se: float, margin: float) -> dict:
    """Two one-sided t-tests (normal approximation) for equivalence."""
    if se == 0:
        return {"t_lower": float("nan"), "t_upper": float("nan"),
                "p_lower": float("nan"), "p_upper": float("nan"),
                "p_tost": float("nan"), "equivalent_at_0.05": False}
    t_lower = (mean_diff - (-margin)) / se
    t_upper = (margin - mean_diff) / se
    # normal-approx one-sided p-values (large-n cohort, se already accounts
    # for clustering when se is the clustered bootstrap SD)
    p_lower = 1 - 0.5 * (1 + math.erf(t_lower / math.sqrt(2)))
    p_upper = 1 - 0.5 * (1 + math.erf(t_upper / math.sqrt(2)))
    p_tost = max(p_lower, p_upper)
    return {
        "t_lower": t_lower, "t_upper": t_upper,
        "p_lower": p_lower, "p_upper": p_upper,
        "p_tost": p_tost, "equivalent_at_0.05": p_tost < 0.05,
    }


def naive_bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    return diffs


def cluster_bootstrap_mean_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample (factory_id, worker_id) clusters (not poses) with replacement,
    per R-CD3.

    `worker_id` is locally scoped per factory in this dataset (each factory
    restarts numbering at 1), so grouping by `worker_id` alone silently
    merges distinct people across factories who happen to share a small
    integer id. The cluster key must be the (factory_id, worker_id) pair.
    """
    workers_a = df_a["_worker_key"].unique()
    workers_b = df_b["_worker_key"].unique()
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        wa = rng.choice(workers_a, size=len(workers_a), replace=True)
        wb = rng.choice(workers_b, size=len(workers_b), replace=True)
        sa = np.concatenate([df_a.loc[df_a["_worker_key"] == w, "rula_grand"].values for w in wa])
        sb = np.concatenate([df_b.loc[df_b["_worker_key"] == w, "rula_grand"].values for w in wb])
        diffs[i] = sa.mean() - sb.mean()
    return diffs


def ci95(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(CSV_PATH)
    df["_domain"] = df.apply(domain_of, axis=1)
    df["_worker_key"] = df["factory_id"].astype(str) + "_" + df["worker_id"].astype(str)

    manu = df[df["_domain"] == "manufacturing"].copy()
    elder = df[df["_domain"] == "eldercare"].copy()

    poses_per_worker = {
        "manufacturing": manu.groupby("_worker_key").size().to_dict(),
        "eldercare": elder.groupby("_worker_key").size().to_dict(),
    }
    n_workers = {
        "manufacturing": int(manu["_worker_key"].nunique()),
        "eldercare": int(elder["_worker_key"].nunique()),
    }

    mean_diff = float(elder["rula_grand"].mean() - manu["rula_grand"].mean())

    naive_diffs = naive_bootstrap_mean_diff(elder["rula_grand"].values, manu["rula_grand"].values, rng)
    naive_se = float(naive_diffs.std(ddof=1))
    naive_ci = ci95(naive_diffs)
    naive_tost = tost(mean_diff, naive_se, MARGIN)

    cluster_diffs = cluster_bootstrap_mean_diff(elder, manu, rng)
    cluster_se = float(cluster_diffs.std(ddof=1))
    cluster_ci = ci95(cluster_diffs)
    cluster_tost = tost(mean_diff, cluster_se, MARGIN)

    metrics = {
        "method": "RULA grand score, eldercare vs manufacturing, E-CD3 margin +/-0.5",
        "cluster_key_note": (
            "Clustered on (factory_id, worker_id). An earlier version of this "
            "script clustered on worker_id alone, which is locally re-numbered "
            "per factory in this dataset and merged distinct workers across "
            "factories who share a small integer id (e.g. F1 worker 1 and F2 "
            "worker 1 are different people). That version undercounted "
            "manufacturing clusters as 12 instead of the true 38."
        ),
        "n_workers": n_workers,
        "poses_per_worker": poses_per_worker,
        "mean_diff_eldercare_minus_manufacturing": mean_diff,
        "naive_pose_level_bootstrap": {
            "n_boot": N_BOOT, "se": naive_se, "ci95": naive_ci, "tost": naive_tost,
        },
        "worker_clustered_bootstrap": {
            "n_boot": N_BOOT, "se": cluster_se, "ci95": cluster_ci, "tost": cluster_tost,
        },
        "ci_width_inflation_clustered_over_naive": (
            (cluster_ci[1] - cluster_ci[0]) / (naive_ci[1] - naive_ci[0])
            if (naive_ci[1] - naive_ci[0]) else float("inf")
        ),
        "interpretation": (
            "Eldercare arm is 2 workers (n=485, n=521 poses); worker-clustered "
            "bootstrap resamples only 2 clusters, so its CI reflects between-worker "
            "variance estimated from n=2 and is not a stable basis for a population "
            "equivalence claim. Reported for R-CD3 delta-disclosure; not proposed as "
            "a defensible equivalence test for eldercare at large."
        ),
    }

    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
