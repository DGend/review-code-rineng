# PWW-DISCLOSURE: public
"""exp-011: worker-clustered bootstrap CI on the C4 calibration report
(Table 3 ECE/Brier), per R-CD3 ("cluster bootstrap by worker_id for
all CD1/CD4 CIs; delta between naive and clustered CIs must be
published"). This does not refit temperature (frozen at T=5.0 for
every domain/method per c4_temperatures.json, exp-005); it only
re-evaluates ECE/Brier on worker-resampled val sets using that frozen
calibrator, so the comparison is apples-to-apples with Table 3.

Reuses the exact record-building and stratified-split logic of
run_c4_calibration.py (same seed=0, val_frac=0.2) so the val set here
is identical to the one Table 3 was computed on; worker_id is carried
alongside as a parallel array (the frozen pipeline does not expose it)
so that cluster resampling stays faithful to R-CD3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path as _Path; _ROOT = _Path(__file__).resolve().parents[4]  # project root
sys.path.insert(0, str(_ROOT))

from ace_pose.confidence_mc import RiskDistribution
from ace_pose.domain_calibration import TemperatureCalibrator, evaluate_calibration

BATCH_DIR = Path(str(_ROOT) + "/ace_pose/_runs/batch_v1")
METHOD_RANGE = {"owas": (1, 4), "rula": (1, 7), "reba": (1, 15)}
PROB_KEY = {"owas": "owas_probs", "rula": "rula_probs", "reba": "reba_probs"}
LABEL_COL = {"owas": "owas_action", "rula": "rula_grand", "reba": "reba_grand"}
VAL_FRAC = 0.2
SPLIT_SEED = 0
N_BOOT = 1000
BOOT_SEED = 20260821


def build_records(df: pd.DataFrame, npz) -> tuple[list, list]:
    """worker_id is locally re-numbered per factory in this dataset (each
    factory restarts at 1), so the cluster key must be the composite
    (factory_id, worker_id) pair, not worker_id alone -- otherwise distinct
    workers across factories who share a small integer id get merged into
    one fake cluster (confirmed: 12 fake vs 38 true manufacturing clusters).
    """
    file_stems = npz["file_stem"]
    stem_to_row = {s: i for i, s in enumerate(file_stems)}
    records, worker_keys = [], []
    for _, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        i = stem_to_row.get(row["file_stem"])
        if i is None:
            continue
        domain = row["domain"]
        worker_key = f"{row.get('factory_id')}_{row.get('worker_id')}"
        for method, (lo, hi) in METHOD_RANGE.items():
            probs = npz[PROB_KEY[method]][i]
            if probs.sum() < 1e-6:
                continue
            label = row[LABEL_COL[method]]
            if pd.isna(label):
                continue
            classes = np.arange(lo, hi + 1, dtype=int)
            dist = RiskDistribution(method=method, classes=classes,
                                     probs=probs / probs.sum(),
                                     n_samples=int(row.get("mc_n_effective", 0)))
            records.append((domain, dist, int(label)))
            worker_keys.append(worker_key)
    return records, worker_keys


def stratified_split_idx(records: list, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (domain, dist, _) in enumerate(records):
        groups.setdefault((domain, dist.method), []).append(i)
    train_idx, val_idx = [], []
    for ids in groups.values():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_idx.extend(ids[:n_val])
        train_idx.extend(ids[n_val:])
    return train_idx, val_idx


def ece_brier_for_idx(records: list, idx: list[int], calibrator: TemperatureCalibrator) -> dict:
    subset = [records[i] for i in idx]
    reports = evaluate_calibration(subset, calibrator)
    return {f"{d}/{m}": {"n": r.n, "brier_before": r.brier_before, "brier_after": r.brier_after,
                          "ece_before": r.ece_before, "ece_after": r.ece_after}
            for (d, m), r in reports.items()}


def main() -> None:
    df = pd.read_csv(BATCH_DIR / "summary_with_mc.csv")
    npz = np.load(BATCH_DIR / "mc_distributions.npz", allow_pickle=True)
    records, worker_keys = build_records(df, npz)
    worker_keys = np.array(worker_keys)

    _, val_idx = stratified_split_idx(records, VAL_FRAC, SPLIT_SEED)
    val_idx = np.array(val_idx)
    val_workers = worker_keys[val_idx]

    temps = json.loads((BATCH_DIR / "c4_temperatures.json").read_text())
    calibrator = TemperatureCalibrator()
    for k, T in temps.items():
        d, m = k.split("/")
        calibrator.by_key[(d, m)] = float(T)

    point = ece_brier_for_idx(records, list(val_idx), calibrator)

    rng = np.random.default_rng(BOOT_SEED)
    # naive: resample val rows directly
    naive_ece_after = {k: [] for k in point}
    for _ in range(N_BOOT):
        resample = rng.choice(val_idx, size=len(val_idx), replace=True)
        rep = ece_brier_for_idx(records, list(resample), calibrator)
        for k in point:
            if k in rep:
                naive_ece_after[k].append(rep[k]["ece_after"])

    # worker-clustered: resample worker_id within domain, take all their val rows
    domain_workers = {}
    for w, wid_domain in zip(val_workers, [records[i][0] for i in val_idx]):
        domain_workers.setdefault(wid_domain, set()).add(w)

    cluster_ece_after = {k: [] for k in point}
    for _ in range(N_BOOT):
        chosen_idx = []
        for domain, workers in domain_workers.items():
            workers = sorted(workers)  # set order varies with string hash seed
            picked = rng.choice(workers, size=len(workers), replace=True)
            for w in picked:
                mask = (val_workers == w) & (np.array([records[i][0] for i in val_idx]) == domain)
                chosen_idx.extend(val_idx[mask].tolist())
        if not chosen_idx:
            continue
        rep = ece_brier_for_idx(records, chosen_idx, calibrator)
        for k in point:
            if k in rep:
                cluster_ece_after[k].append(rep[k]["ece_after"])

    def ci95(x: list[float]) -> list[float] | None:
        if len(x) < 10:
            return None
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {}
    for k in point:
        naive_ci = ci95(naive_ece_after[k])
        cluster_ci = ci95(cluster_ece_after[k])
        width_ratio = None
        if naive_ci and cluster_ci and (naive_ci[1] - naive_ci[0]) > 0:
            width_ratio = (cluster_ci[1] - cluster_ci[0]) / (naive_ci[1] - naive_ci[0])
        out[k] = {
            "point": point[k],
            "n_val_workers": len(domain_workers.get(k.split("/")[0], [])),
            "naive_bootstrap_ece_after_ci95": naive_ci,
            "worker_clustered_bootstrap_ece_after_ci95": cluster_ci,
            "ci_width_inflation_clustered_over_naive": width_ratio,
        }

    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
