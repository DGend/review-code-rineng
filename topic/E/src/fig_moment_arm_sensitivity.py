# PWW-DISCLOSURE: public
"""Sensitivity of the NIOSH AL/MPL exceedance rates to the erector-spinae
moment arm (Chaffin single-equivalent-muscle model, Section 3.6), which
the manuscript uses at a single fixed value (0.05 m). Varies it +/-20%
(0.04 m, 0.06 m) and reports how the cohort-level exceedance
percentages shift. Reuses each pose's already-computed dense-spine
moments; no pose re-inference is needed, so this is fast.

Source data (real, not fabricated):
  ace_pose/_runs/batch_v1/summary_with_xw.csv (n=6,007, status==ok)
  ace_pose/_runs/batch_v1/details/*.npz (pose keypoints)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import ace_pose.disc_load as disc_load
from ace_pose.dense_spine import estimate_dense_spine
from ace_pose.disc_load import trunk_flexion_deg_3d, NIOSH_AL_N, NIOSH_MPL_N

BATCH = ROOT / "ace_pose/_runs/batch_v1"
BODY_MASS_KG = 70.0
MOMENT_ARMS = [0.04, 0.05, 0.06]


def main():
    for cand in ("summary_with_xw.csv", "summary_with_mc.csv", "summary_with_pred.csv", "summary.csv"):
        p = BATCH / cand
        if p.exists():
            sum_path = p
            break
    df = pd.read_csv(sum_path)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    det = BATCH / "details"

    poses = []
    for _, r in df.iterrows():
        npz = det / f"{r['file_stem']}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        ls = float(r["biomech_length_scale_m"])
        if not np.isfinite(ls) or ls <= 0:
            continue
        poses.append((d["kpts_m25"], d["conf_m25"], ls))
    print(f"loaded {len(poses)} poses")

    results = {}
    for arm in MOMENT_ARMS:
        disc_load.ERECTOR_MOMENT_ARM_M = arm  # module-level monkeypatch
        n_al = n_mpl = n_total = 0
        for kp, cf, ls in poses:
            spine = estimate_dense_spine(kp, cf, body_mass_kg=BODY_MASS_KG, length_scale_m=ls)
            flex = trunk_flexion_deg_3d(kp, cf)
            comp = disc_load.disc_compression_N(spine, BODY_MASS_KG, flex)
            c = comp.get("L5/S1", 0.0)
            n_total += 1
            if c >= NIOSH_AL_N:
                n_al += 1
            if c >= NIOSH_MPL_N:
                n_mpl += 1
        results[arm] = (100 * n_al / n_total, 100 * n_mpl / n_total)
        print(f"moment arm {arm:.2f} m: AL-exceed {100*n_al/n_total:.1f}%  MPL-exceed {100*n_mpl/n_total:.1f}%  (n={n_total})")

    base_al, base_mpl = results[0.05]
    print("\n=== +/-20% summary vs base (0.05 m) ===")
    for arm in MOMENT_ARMS:
        al, mpl = results[arm]
        print(f"{arm:.2f} m: AL {al:.1f}% (delta {al-base_al:+.1f} pp), MPL {mpl:.1f}% (delta {mpl-base_mpl:+.1f} pp)")


if __name__ == "__main__":
    main()
