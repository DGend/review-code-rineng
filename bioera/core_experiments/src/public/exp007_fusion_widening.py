# PWW-DISCLOSURE: public
"""exp-007 fusion widening feasibility audit and R7 fallback trigger.

R7 (from paper/goal.md): Table 11 fusion-mode comparison is on n = 20
high-risk worker poses; widen to n >= 100 multi-worker frames, or if
not feasible, restate C5 as a case-study finding.

exp-007 audits feasibility on the frozen batch_v1 artefacts and
records the outcome deterministically. It does NOT re-run the
multi-backbone ensemble.

Facts recorded:
1. Multi-worker frame availability (n_detections >= 2) in the full
   cohort - the pool that a widened fusion study could draw from.
2. Whether per-backbone (Body / BodyWithFeet / Wholebody3d)
   alternative predictions are persisted in the batch_v1 detail
   npz files - required to widen without re-running the ensemble.
3. Feasibility verdict and the R7 branch actually taken.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find(rel: str) -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(rel)


def main() -> None:
    xw = _find("ace_pose/_runs/batch_v1/summary_with_xw.csv")
    df = pd.read_csv(xw)
    n_total = int(len(df))
    n_multi = int((df["n_detections"] >= 2).sum())
    n_multi_ge3 = int((df["n_detections"] >= 3).sum())
    per_dom = {
        dom: int(((sub["n_detections"] >= 2)).sum())
        for dom, sub in df.groupby("domain")
    }

    details_dir = _find("ace_pose/_runs/batch_v1/details")
    # Stratified probe: for every unique (domain, factory, process, action,
    # worker) tuple in the cohort, sample up to N files. This defends
    # against a same-recording sampling artefact.
    strata_cols = ["domain", "factory_id", "process_id", "action_id", "worker_id"]
    strata_present = [c for c in strata_cols if c in df.columns]
    per_stratum_cap = 2
    picks: list[str] = []
    for _, grp in df.groupby(strata_present):
        picks.extend(grp["file_stem"].head(per_stratum_cap).tolist())
    sampled_stems = sorted(set(picks))
    samples: list[Path] = []
    for stem in sampled_stems:
        p = details_dir / f"{stem}.npz"
        if p.exists():
            samples.append(p)
    detected_keys: set[str] = set()
    for p in samples:
        with np.load(p, allow_pickle=True) as npz:
            detected_keys.update(npz.files)

    required_alt_keys = {
        "kpts_body_17",
        "kpts_bodywithfeet_23",
        "kpts_wholebody3d_133_alt",
        "conf_per_backbone",
    }
    alt_keys_present = sorted(detected_keys & required_alt_keys)
    alt_keys_missing = sorted(required_alt_keys - detected_keys)

    feasible = len(alt_keys_present) > 0
    r7_branch = "widen" if feasible else "restate-as-case-study"

    metrics = {
        "cohort_n_total": n_total,
        "multi_worker_pool_n_ge2": n_multi,
        "multi_worker_pool_n_ge3": n_multi_ge3,
        "multi_worker_pool_frac": n_multi / n_total,
        "per_domain_multi_ge2": per_dom,
        "detail_npz_sampled_n": len(samples),
        "detail_npz_strata_cols": strata_present,
        "detail_npz_unique_strata": int(df.groupby(strata_present).ngroups) if strata_present else 0,
        "detail_npz_per_stratum_cap": per_stratum_cap,
        "detail_npz_first_5_sampled": [p.name for p in samples[:5]],
        "detail_npz_last_5_sampled": [p.name for p in samples[-5:]],
        "detail_npz_keys_sampled": sorted(detected_keys),
        "required_alt_keys": sorted(required_alt_keys),
        "alt_keys_present": alt_keys_present,
        "alt_keys_missing": alt_keys_missing,
        "widening_feasible_without_rerun": feasible,
        "r7_branch_taken": r7_branch,
        "target_n_min": 100,
    }

    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
