# PWW-DISCLOSURE: public
"""exp-012: refit C4 per-(domain, method) temperature under a
worker-held-out split (R-CD3 full requirement), replacing the
pose-level 80/20 split that exp-005/run_c4_calibration.py used, which
exp-011 found leaks worker identity between train and val (every val
worker also appears in train).

Cluster key is (factory_id, worker_id) per the exp-010/011 bugfix
(worker_id is locally re-numbered per factory).

Per-domain worker-holdout strategy (declared, not tuned to the data):
  - manufacturing (many workers): random 80/20 GROUP split -- whole
    workers assigned to train or val, target ~20% of poses in val,
    repeated N_MANUFACTURING_REPEATS times.
  - eldercare and logistics (a handful of workers each): leave-one-worker-
    out over every worker, each fold reported with its validation size,
    plus the mean and range across folds.
worker_id is the labelled worker matched to the analysed person (at the
eldercare and logistics sites the file-name W token is a recording session).
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
SEED = 20260822
VAL_FRAC_TARGET = 0.2


def build_records(df: pd.DataFrame, npz) -> tuple[list, list]:
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
    return records, np.array(worker_keys)


N_MANUFACTURING_REPEATS = 20


def group_split_manufacturing(domain_idx: list[int], worker_keys: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Random group (worker) split targeting ~20% of poses in val.

    One worker (F1 worker 1) alone contributes 1,151 of 4,001
    manufacturing poses (29%), larger than the 20% val target. A
    single greedy split can therefore land on a degenerate 1-worker
    val fold purely by shuffle luck. Caller must average over many
    repeats (N_MANUFACTURING_REPEATS) rather than trust one split.
    """
    workers = list(dict.fromkeys(worker_keys[domain_idx]))
    rng.shuffle(workers)
    counts = {w: int((worker_keys[domain_idx] == w).sum()) for w in workers}
    total = sum(counts.values())
    target = total * VAL_FRAC_TARGET
    val_workers, running = set(), 0
    for w in workers:
        # stop adding once val is within [target, target*1.5]; skip a
        # worker that would overshoot past 1.5x target unless nothing
        # has been added yet (guarantees non-empty val).
        if running >= target:
            break
        if running + counts[w] > target * 1.5 and val_workers:
            continue
        val_workers.add(w)
        running += counts[w]
    val_idx = [i for i in domain_idx if worker_keys[i] in val_workers]
    train_idx = [i for i in domain_idx if worker_keys[i] not in val_workers]
    return train_idx, val_idx


def report_for_split(records: list, train_idx: list[int], val_idx: list[int]) -> dict:
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    cal = TemperatureCalibrator().fit(train)
    base = TemperatureCalibrator()
    val_reports = evaluate_calibration(val, cal)
    base_reports = evaluate_calibration(val, base)
    out = {}
    for key, rep in val_reports.items():
        d, m = key
        out[f"{d}/{m}"] = {
            "n_train": sum(1 for i in train_idx if records[i][0] == d and records[i][1].method == m),
            "n_val": rep.n,
            "T": cal.get_T(d, m),
            "brier_before": base_reports[key].brier_before,
            "brier_after": rep.brier_after,
            "ece_before": base_reports[key].ece_before,
            "ece_after": rep.ece_after,
        }
    return out


def main() -> None:
    df = pd.read_csv(BATCH_DIR / "summary_with_mc.csv")
    npz = np.load(BATCH_DIR / "mc_distributions.npz", allow_pickle=True)
    records, worker_keys = build_records(df, npz)
    domains = np.array([r[0] for r in records])
    rng = np.random.default_rng(SEED)

    result = {}

    # Manufacturing: repeated random group (worker) 80/20 splits,
    # averaged, because a single split can land on a degenerate
    # 1-worker val fold (see group_split_manufacturing docstring).
    manu_idx = [i for i, d in enumerate(domains) if d == "manufacturing"]
    repeats = []
    for _ in range(N_MANUFACTURING_REPEATS):
        m_train, m_val = group_split_manufacturing(manu_idx, worker_keys, rng)
        rep = report_for_split(records, m_train, m_val)
        rep["_n_val_workers"] = len(set(worker_keys[m_val]))
        rep["_n_train_workers"] = len(set(worker_keys[m_train]))
        repeats.append(rep)

    agg = {}
    for method_key in ["manufacturing/owas", "manufacturing/rula", "manufacturing/reba"]:
        eces = [r[method_key]["ece_after"] for r in repeats if method_key in r]
        briers = [r[method_key]["brier_after"] for r in repeats if method_key in r]
        agg[method_key] = {
            "n_repeats": len(eces),
            "ece_after_mean": float(np.mean(eces)),
            "ece_after_std": float(np.std(eces)),
            "ece_after_min_max": [float(np.min(eces)), float(np.max(eces))],
            "brier_after_mean": float(np.mean(briers)),
        }
    result["manufacturing_worker_held_out_repeated"] = agg
    result["manufacturing_worker_held_out_single_split_examples"] = repeats[:3]
    result["manufacturing_n_val_workers_range"] = [
        min(r["_n_val_workers"] for r in repeats), max(r["_n_val_workers"] for r in repeats)
    ]

    # Eldercare and logistics: leave-one-worker-out over every worker.
    for dom in ("eldercare", "logistics"):
        idx = [i for i, d in enumerate(domains) if d == dom]
        folds = {}
        for held_out in sorted(set(worker_keys[idx])):
            val_idx = [i for i in idx if worker_keys[i] == held_out]
            train_idx = [i for i in idx if worker_keys[i] != held_out]
            folds[held_out] = report_for_split(records, train_idx, val_idx)
        summary = {}
        for m in METHOD_RANGE:
            eces = [f[f"{dom}/{m}"]["ece_after"] for f in folds.values() if f"{dom}/{m}" in f]
            summary[f"{dom}/{m}"] = {"n_folds": len(eces), "ece_after_mean": float(np.mean(eces)),
                                     "ece_after_min_max": [float(np.min(eces)), float(np.max(eces))]}
        result[f"{dom}_leave_one_worker_out"] = {"folds": folds, "summary": summary}

    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
