# PWW-DISCLOSURE: public
"""3D skeletons of cohort workers from the range-of-motion-constrained fit (Supplementary S9).

For each example image: the image with the worker's 2D keypoints (face pixelated first), and the
fitted 3D skeleton drawn in perspective on a floor grid, from an oblique and from a side
viewpoint (heading removed, y up). Every solution
the fit keeps is drawn, the best in colour and the others in grey, so a pose the image leaves
ambiguous shows as diverging grey skeletons. The 3D joints of every kept solution are saved as
JSON next to the figure, so the reconstruction can be reused.

Usage: python topic/E/src/fig_lift3d_examples.py [FILE_STEM ...]
  (default: the first sample image of F1 manufacturing, F6 eldercare, F7 logistics)
Writes topic/E/paper/figures/supplementary/S9.png and topic/E/result/lift3d_examples/<stem>.json.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose import lift3d_rom as L  # noqa: E402
from ace_pose.data_paths import IMG_DIR  # noqa: E402
from ace_pose.ergo_v2 import angles as A  # noqa: E402
from ace_pose.major_joints import Major25Pose, coco_wholebody_to_major25  # noqa: E402
from ace_pose.run_pose_visualization import mosaic_face  # noqa: E402
from carda_angle_eval import frame_3d, hands_2d, measure  # noqa: E402

BATCH = ROOT / "ace_pose/_runs/batch_v1"
FIG = ROOT / "topic/E/paper/figures/supplementary/S9.png"
OUT = ROOT / "topic/E/result/lift3d_examples"
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
         (15, 17), (16, 20), (23, 0)]
LEFT = {(5, 7), (7, 9), (11, 13), (13, 15), (15, 17)}
DOMAIN = {"F1": "Manufacturing", "F6": "Eldercare", "F7": "Logistics"}


def default_stems():
    idx = pd.read_csv(ROOT / "topic/E/result/pose_model_comparison/samples/index.csv")
    return [idx[idx.factory == f].file_stem.iloc[0] for f in ("F1", "F6", "F7")]


def draw3d(ax, j, color, lw, alpha, joints=False):
    """Skeleton on a 3D axis: x (subject's left) -> X, z (forward) -> Y, y (up) -> Z."""
    X, Y, Z = j[:, 0], j[:, 2], j[:, 1]
    for a, b in EDGES:
        c = color if color != "auto" else ("tab:blue" if (a, b) in LEFT else "tab:red")
        ax.plot([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]], color=c, lw=lw, alpha=alpha, solid_capstyle="round")
    if joints:
        used = sorted({i for e in EDGES for i in e})
        ax.scatter(X[used], Y[used], Z[used], s=14, c="k", depthshade=True)


def setup3d(ax, j, elev, azim, title):
    floor = j[:, 1].min()
    g = np.linspace(-0.5, 0.5, 6)
    for v in g:
        ax.plot([v, v], [-0.5, 0.5], [floor, floor], color="0.85", lw=0.6)
        ax.plot([-0.5, 0.5], [v, v], [floor, floor], color="0.85", lw=0.6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(floor, floor + 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main(stems):
    OUT.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(BATCH / "summary.csv").set_index("file_stem")
    fig = plt.figure(figsize=(11, 3.9 * len(stems)))
    gs = fig.add_gridspec(len(stems), 3, width_ratios=[1.1, 1.2, 1.2])
    axes = [[fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1], projection="3d"),
             fig.add_subplot(gs[r, 2], projection="3d")] for r in range(len(stems))]
    up = np.array([0.0, 1.0, 0.0])
    for row, stem in zip(axes, stems):
        z = np.load(BATCH / "details" / f"{stem}.npz")
        k2, s = z["kpts2d_133"].astype(float), z["scores_133"].astype(float)
        m25 = coco_wholebody_to_major25(np.c_[k2, np.zeros(133)], s)
        fit = L.fit(m25.keypoints[:, :2], m25.confidence, hands2d=hands_2d(k2, s)[0], hands_conf=hands_2d(k2, s)[1])
        best = int(np.argmin(fit.rms))
        meas = []
        for j, hd in zip(fit.body_joints, fit.hand_joints):
            p = Major25Pose(keypoints=j, confidence=np.ones(25), source=np.zeros(25, np.int8))
            meas.append(measure(p, frame_3d(p, up), up, hd))
        (OUT / f"{stem}.json").write_text(json.dumps({
            "file_stem": stem, "coordinates": "stature = 1, y up, z = pelvis heading, x = subject's left",
            "joint_order": "major-25 (ace_pose/major_joints.py)", "best": best,
            "solutions": [{"rms": r, "joints": j.round(4).tolist(), "hand_knuckles": h.round(4).tolist(), "angles": a}
                          for r, j, h, a in zip(fit.rms, fit.body_joints, fit.hand_joints, meas)]}, indent=1))
        # image with 2D keypoints, face pixelated before anything is drawn
        img = mosaic_face(cv2.imread(str(IMG_DIR / f"{stem}.jpg")), k2, s)
        kp = m25.keypoints[:, :2]
        ok = m25.confidence >= 0.3
        x0, y0 = kp[ok].min(0) - 60
        x1, y1 = kp[ok].max(0) + 60
        h, w = img.shape[:2]
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        row[0].imshow(crop)
        k = kp - [x0, y0]
        for a, b in EDGES:
            if ok[a] and ok[b]:
                row[0].plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], color="tab:blue" if (a, b) in LEFT else "tab:red", lw=2)
        row[0].axis("off")
        f = summ.at[stem, "factory_id"] if "factory_id" in summ.columns else ""
        row[0].set_title(f"{DOMAIN.get(f'F{int(f)}', '')} image, 2D keypoints", fontsize=9)
        jb = fit.body_joints[best]
        for ax, elev, azim, name in ((row[1], 18, -60, "3D, oblique view"), (row[2], 8, 0, "3D, side view")):
            setup3d(ax, jb, elev, azim, name)
            for i, j in enumerate(fit.body_joints):
                if i != best:
                    draw3d(ax, j, "0.6", 1.2, 0.35)
            draw3d(ax, jb, "auto", 2.6, 1.0, joints=True)
        mb = meas[best]
        tf = [d["trunk_flexion"] for d in meas]
        row[2].text2D(0.02, 0.0, f"trunk flexion {mb['trunk_flexion']:.0f} deg (kept: {min(tf):.0f} to {max(tf):.0f})\n"
                    f"upper-arm abduction R {mb['upper_arm_abduction_r']:.0f}, L {mb['upper_arm_abduction_l']:.0f} deg\n"
                    f"{len(fit.body_joints)} solution(s) kept", transform=row[2].transAxes, fontsize=7, va="bottom")
        print(f"{stem}: {len(fit.body_joints)} solutions, trunk flexion {mb['trunk_flexion']:.0f} ({min(tf):.0f}-{max(tf):.0f})")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    print(f"-> {FIG}")


if __name__ == "__main__":
    main(sys.argv[1:] or default_stems())
