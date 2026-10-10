"""Apply C3 confidence-aware Monte Carlo scoring to every image in a batch.

Reads `summary_with_pred.csv` plus per-image `details/*.npz`, runs MC on
each pose, and writes:
  - `summary_with_mc.csv` — adds MC mean / variance / entropy / mode for
    each method
  - `mc_distributions.npz` — full distributional probability vectors
    (shape (N, 4) for OWAS, (N, 7) for RULA, (N, 15) for REBA) — used
    by C4 calibration downstream
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.ergo_v2.context_policy import cohort_context  # posture only, see context_policy
from ace_pose.confidence_mc import McConfig, mc_score_all
from ace_pose.major_joints import Major25Pose

log = logging.getLogger("ace_pose.run_c3_mc")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _summarize_dist(dist):
    return {
        "mode": dist.mode,
        "expected": dist.expected,
        "variance": dist.variance,
        "entropy": dist.entropy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--sigma_fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N images (smoke test)")
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    sum_path = batch_dir / "summary_with_pred.csv"
    if not sum_path.exists():
        sum_path = batch_dir / "summary.csv"
    df = pd.read_csv(sum_path)
    detail_dir = batch_dir / "details"

    if "status" in df.columns:
        df = df.loc[df["status"] == "ok"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    log.info("Running C3 MC on %d images (n_samples=%d, sigma_fraction=%.3f)",
             len(df), args.n_samples, args.sigma_fraction)

    cfg = McConfig(
        n_samples=args.n_samples,
        sigma_fraction=args.sigma_fraction,
        rng_seed=args.seed,
    )

    world_up = np.array([0.0, -1.0, 0.0])

    owas_probs = np.zeros((len(df), 4), dtype=np.float32)
    rula_probs = np.zeros((len(df), 7), dtype=np.float32)
    reba_probs = np.zeros((len(df), 15), dtype=np.float32)
    out_records = []

    t_start = time.time()
    last_log = t_start
    for i, row in enumerate(df.itertuples()):
        rec = {"file_stem": row.file_stem}
        npz_path = detail_dir / f"{row.file_stem}.npz"
        if not npz_path.exists():
            out_records.append(rec)
            continue
        data = np.load(npz_path)
        kp = data["kpts_m25"]
        conf = data["conf_m25"]
        pose = Major25Pose(
            keypoints=kp.astype(np.float64),
            confidence=conf.astype(np.float64),
            source=np.zeros(25, dtype=np.int8),
        )
        try:
            res = mc_score_all(
                pose, config=cfg, world_up=world_up,
                walking=False, **cohort_context(getattr(row, "process_label", None)),
                domain=row.domain,
                process_id=int(getattr(row, "process_id", 0)) or None,
            )
        except Exception as ex:
            rec["mc_error"] = str(ex)
            out_records.append(rec)
            continue
        # store distributions
        for c in res.owas_dist.classes:
            owas_probs[i, c - 1] = res.owas_dist.probs[c - res.owas_dist.classes[0]]
        for c in res.rula_dist.classes:
            rula_probs[i, c - 1] = res.rula_dist.probs[c - res.rula_dist.classes[0]]
        for c in res.reba_dist.classes:
            reba_probs[i, c - 1] = res.reba_dist.probs[c - res.reba_dist.classes[0]]

        rec.update({
            "mc_owas_mode": res.owas_dist.mode,
            "mc_owas_expected": res.owas_dist.expected,
            "mc_owas_variance": res.owas_dist.variance,
            "mc_owas_entropy": res.owas_dist.entropy,
            "mc_rula_mode": res.rula_dist.mode,
            "mc_rula_expected": res.rula_dist.expected,
            "mc_rula_variance": res.rula_dist.variance,
            "mc_rula_entropy": res.rula_dist.entropy,
            "mc_reba_mode": res.reba_dist.mode,
            "mc_reba_expected": res.reba_dist.expected,
            "mc_reba_variance": res.reba_dist.variance,
            "mc_reba_entropy": res.reba_dist.entropy,
            "mc_n_effective": res.owas_dist.n_samples,
        })
        out_records.append(rec)

        now = time.time()
        if now - last_log > 5.0 or i == len(df) - 1:
            done = i + 1
            rate = done / (now - t_start + 1e-9)
            eta = (len(df) - done) / max(rate, 1e-6)
            log.info("%4d/%d  %.1f img/s  ETA %.0fs",
                     done, len(df), rate, eta)
            last_log = now

    log.info("Done in %.1fs.", time.time() - t_start)

    mc_df = pd.DataFrame(out_records)
    merged = df.merge(mc_df, on="file_stem", how="left")
    out_csv = batch_dir / "summary_with_mc.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(merged))

    out_npz = batch_dir / "mc_distributions.npz"
    np.savez_compressed(
        out_npz,
        file_stem=df["file_stem"].to_numpy(),
        owas_probs=owas_probs,
        rula_probs=rula_probs,
        reba_probs=reba_probs,
    )
    log.info("Wrote %s", out_npz)


if __name__ == "__main__":
    main()
