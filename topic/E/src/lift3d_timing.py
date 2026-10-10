# PWW-DISCLOSURE: public
"""CPU time of the 3D range-of-motion fit (ace_pose/lift3d_rom.py) per pose, and the total per
image when it follows each compared pose model (GPU times from accuracy_speed.csv) and the
scoring layer. One process, one thread (OMP_NUM_THREADS=1), every 100th cohort pose (60 poses).

Usage: OMP_NUM_THREADS=1 python topic/E/src/lift3d_timing.py
Writes topic/E/result/pose_model_comparison/lift3d_timing.csv.
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose import lift3d_rom as L  # noqa: E402
from ace_pose.major_joints import coco_wholebody_to_major25  # noqa: E402
from carda_angle_eval import hands_2d  # noqa: E402

OUT = ROOT / "topic/E/result/pose_model_comparison"
SCORING_MS = 0.4          # scoring layer per pose (compare_pose_models_speed.py, "scoring")


def main():
    jobs = []
    for f in sorted(glob.glob(str(ROOT / "ace_pose/_runs/batch_v1/details/*.npz")))[::100][:60]:
        z = np.load(f)
        m = coco_wholebody_to_major25(np.c_[z["kpts2d_133"], np.zeros(133)], z["scores_133"])
        jobs.append((m.keypoints[:, :2], m.confidence, *hands_2d(z["kpts2d_133"], z["scores_133"])))
    fit = {}
    for name, kw in (("8 starting headings (as evaluated)", {}), ("4 starting headings", {"headings": range(0, 360, 90)})):
        t = []
        for kp, c, h, hc in jobs:
            t0 = time.perf_counter()
            L.fit(kp, c, hands2d=h, hands_conf=hc, **kw)
            t.append((time.perf_counter() - t0) * 1e3)
        fit[name] = (float(np.median(t)), float(np.percentile(t, 90)))
        print(f"3D fit, {name}: median {fit[name][0]:.0f} ms, p90 {fit[name][1]:.0f} ms (n={len(t)})")
    sp = pd.read_csv(OUT / "accuracy_speed.csv")
    lift = fit["8 starting headings (as evaluated)"][0]
    rows = [{"model": r.model, "pose_gpu_ms": r.gpu_ms, "scoring_ms": SCORING_MS,
             "image_plane_total_ms": r.gpu_ms + SCORING_MS, "lift3d_cpu_ms_median": lift,
             "with_lift3d_total_ms": r.gpu_ms + lift + SCORING_MS} for r in sp.itertuples()]
    pd.DataFrame(rows).to_csv(OUT / "lift3d_timing.csv", index=False, float_format="%.1f")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
