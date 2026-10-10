# PWW-DISCLOSURE: public
"""Section 4.8 "Context extraction": how much REBA moves when the legacy zero
defaults (coupling Good = 0, no activity flag) are replaced by the conservative
defaults (coupling Fair = 1, static-posture flag on).

Why this exists: the earlier manuscript number (+1.0 point, 41% from level 1
to 2, Wilcoxon p < 1e-50) had no producer script and does not match this
definition; +1.0 is what the static flag alone gives, and that flag adds
exactly one point to every grand score, so a significance test on it is
uninformative.

Source data: ace_pose/_runs/batch_v1/summary.csv + details/*.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.ergo_v2.reba import score_reba
from ace_pose.major_joints import Major25Pose

BATCH = ROOT / "ace_pose/_runs/batch_v1"
VARIANTS = {"legacy zero": dict(coupling=0, activity_static=False),
            "coupling Fair only": dict(coupling=1, activity_static=False),
            "static flag only": dict(coupling=0, activity_static=True),
            "conservative (both)": dict(coupling=1, activity_static=True)}


def main():
    df = pd.read_csv(BATCH / "summary.csv")
    out = {k: [] for k in VARIANTS}
    for stem in df[df.status == "ok"].file_stem:
        d = np.load(BATCH / "details" / f"{stem}.npz")
        pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                           confidence=d["conf_m25"].astype(np.float64),
                           source=np.zeros(25, dtype=np.int8))
        for k, kw in VARIANTS.items():
            r = score_reba(pose, world_up=np.array([0.0, -1.0, 0.0]), **kw)
            out[k].append((r.grand_score, r.action_level))
    base = np.array(out["legacy zero"])
    print(f"n = {len(base)}, legacy mean REBA {base[:, 0].mean():.2f}")
    for k in list(VARIANTS)[1:]:
        v = np.array(out[k])
        print(f"{k:20s} mean diff {(v[:, 0] - base[:, 0]).mean():+.2f}  "
              f"level 1->2 {100*((base[:, 1] == 1) & (v[:, 1] == 2)).mean():.1f}%  "
              f"any upward level shift {100*(v[:, 1] > base[:, 1]).mean():.1f}%")


if __name__ == "__main__":
    main()
