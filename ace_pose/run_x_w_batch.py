"""Apply X (biomechanics) and W (VLM-context heuristic) to every pose in
the batch, adding their outputs as new columns to the summary CSV.

Output: `summary_with_xw.csv` (and `summary_with_mc_xw.csv` if MC summary
is present).
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

from ace_pose.biomechanics import (
    _estimate_stature, compute_static_moments, headline_torques,
)
from ace_pose.major_joints import Major25Pose
from ace_pose.vlm_context import ContextExtractor, HeuristicBackend, context_to_columns


log = logging.getLogger("ace_pose.run_x_w_batch")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--body_mass_kg", type=float, default=70.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--length_scale_m", type=float, default=None,
                        help="multiplier to convert pose distances to metres. "
                             "When omitted, scale is auto-computed per pose so "
                             "stature ≈ 1.72 m (cohort mean).")
    parser.add_argument("--target_stature_m", type=float, default=1.72)
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    # Prefer the most-enriched summary if present
    for cand in ("summary_with_mc.csv", "summary_with_pred.csv", "summary.csv"):
        path = batch_dir / cand
        if path.exists():
            sum_path = path
            break
    log.info("Reading %s", sum_path)
    df = pd.read_csv(sum_path)

    if "status" in df.columns:
        ok_mask = df["status"] == "ok"
    else:
        ok_mask = pd.Series([True] * len(df))
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        ok_mask = ok_mask.iloc[:args.limit].reset_index(drop=True)

    detail_dir = batch_dir / "details"
    extractor = ContextExtractor(HeuristicBackend())
    world_up = np.array([0.0, -1.0, 0.0])

    new_rows = []
    t_start = time.time()
    last_log = t_start
    n_ok = 0
    for i, row in enumerate(df.itertuples()):
        rec = {"file_stem": row.file_stem}
        if not ok_mask.iloc[i]:
            new_rows.append(rec)
            continue
        npz_path = detail_dir / f"{row.file_stem}.npz"
        if not npz_path.exists():
            new_rows.append(rec)
            continue
        data = np.load(npz_path)
        pose = Major25Pose(
            keypoints=data["kpts_m25"].astype(np.float64),
            confidence=data["conf_m25"].astype(np.float64),
            source=np.zeros(25, dtype=np.int8),
        )
        # X biomech — auto length_scale so stature ≈ target
        try:
            if args.length_scale_m is None:
                stature_pose = _estimate_stature(pose.keypoints)
                ls = args.target_stature_m / max(stature_pose, 1e-6)
            else:
                ls = args.length_scale_m
            biomech = compute_static_moments(
                pose, body_mass_kg=args.body_mass_kg,
                length_scale_m=ls, world_up=world_up,
            )
            rec.update(headline_torques(biomech))
            rec["biomech_length_scale_m"] = ls
        except Exception as ex:
            rec["biomech_error"] = f"{type(ex).__name__}: {ex}"
        # W context (heuristic)
        try:
            ctx = extractor(pose=pose)
            rec.update(context_to_columns(ctx))
        except Exception as ex:
            rec["ctx_error"] = f"{type(ex).__name__}: {ex}"

        new_rows.append(rec)
        n_ok += 1

        now = time.time()
        if now - last_log > 5.0 or i == len(df) - 1:
            rate = (i + 1) / (now - t_start + 1e-9)
            log.info("%4d/%d  %.0f img/s", i + 1, len(df), rate)
            last_log = now

    log.info("Done in %.1fs (%d processed)", time.time() - t_start, n_ok)

    out_df = df.merge(pd.DataFrame(new_rows), on="file_stem", how="left")
    out_csv = batch_dir / "summary_with_xw.csv"
    out_df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # Quick summary
    log.info("=" * 60)
    if "biomech_L5S1_Nm" in out_df.columns:
        log.info("L5/S1 torque (Nm) by domain:")
        log.info("\n%s", out_df.groupby("domain")["biomech_L5S1_Nm"]
                 .describe()[["count", "mean", "50%", "max"]].round(2).to_string())
        log.info("Shoulder max torque (Nm) by domain:")
        log.info("\n%s", out_df.groupby("domain")["biomech_shoulder_max_Nm"]
                 .describe()[["count", "mean", "50%", "max"]].round(2).to_string())
    if "ctx_coupling" in out_df.columns:
        log.info("Context-coupling distribution by domain:")
        log.info("\n%s", pd.crosstab(out_df["domain"], out_df["ctx_coupling"]).to_string())
        log.info("Context-load_band distribution by domain:")
        log.info("\n%s", pd.crosstab(out_df["domain"], out_df["ctx_load_band"]).to_string())


if __name__ == "__main__":
    main()
