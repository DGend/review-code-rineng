# PWW-DISCLOSURE: public
"""Supplementary Fig. S4: every compared pose model on the nine domain-by-risk-tier images.

The nine images are the domain (manufacturing, eldercare, logistics) by REBA tier (low, medium,
high) examples of main-text Fig. 4 (ace_pose.run_env_load_visualization.select_load_samples).
Every one of the seven models of Supplementary S7 is run on every image (real inference, no
schematic layouts); the detection closest to the analysed worker (the pipeline's cached 2D
keypoints) is drawn, all 17 COCO body joints (no confidence cut: the models' scores are on
different scales). Faces are pixelated before anything is drawn: the
worker's from the RTMW3D face landmarks, everyone else's from each detection's head keypoints.

Usage (GPU environment with rtmlib, onnxruntime-gpu and, for DETRPose, PyTorch and DETRPOSE_ROOT):
  python topic/E/src/fig_pose_models_grid.py
Writes ace_pose/_runs/batch_v1/analysis/figures/pose_models_grid.png (collect_figures.py copies it
to the paper as Supplementary Fig. S4). Needs the AI-Hub images.
"""
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
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
from ace_pose.data_paths import IMG_DIR  # noqa: E402
from ace_pose.run_env_load_visualization import select_load_samples  # noqa: E402
from ace_pose.run_pose_visualization import mosaic_face  # noqa: E402
from video_pose_eval import build  # noqa: E402

BATCH = ROOT / "ace_pose/_runs/batch_v1"
OUT = BATCH / "analysis/figures/pose_models_grid.png"
MODELS = [("rtmw3d", "RTMW3D-x (used)"), ("rtmw", "RTMW-x"), ("rtmpose", "RTMPose-x"), ("rtmo", "RTMO-l"),
          ("vitpose", "ViTPose-B"), ("detrpose_l", "DETRPose-L"), ("detrpose_x", "DETRPose-X")]
DOMAINS, TIERS = ("manufacturing", "eldercare", "logistics"), ("low", "medium", "high")
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def blur_heads(img, kps, scs, block=12):
    """Pixelate each detected person's head from nose, eyes and ears (scaled by shoulder width)."""
    h, w = img.shape[:2]
    for k, s in zip(kps, scs):
        head = k[:5][s[:5] > 0.1]
        if len(head) == 0:
            continue
        sh = np.linalg.norm(k[5] - k[6]) if min(s[5], s[6]) > 0.1 else 0.0
        half = max(np.ptp(head[:, 0]), np.ptp(head[:, 1]), 0.6 * sh, 20) * 0.9
        cx, cy = head.mean(axis=0)
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
        if x1 > x0 + 4 and y1 > y0 + 4:
            r = img[y0:y1, x0:x1]
            small = cv2.resize(r, (max(1, r.shape[1] // block), max(1, r.shape[0] // block)))
            img[y0:y1, x0:x1] = cv2.resize(small, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img


def nearest(k, ref):
    """Index of the detection whose body keypoints are closest to the worker's (median distance)."""
    if k is None or len(k) == 0:
        return None
    d = [np.median(np.linalg.norm(p[:17] - ref, axis=1)) for p in k]
    return int(np.argmin(d))


def draw(img, k, color=(40, 80, 230)):
    """All 17 body joints: the models' confidence scores are on different scales (RTMW's are not
    in 0-1), so no confidence cut is applied, to keep the panels comparable."""
    t = max(2, img.shape[0] // 250)
    for a, b in EDGES:
        cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), color, t, cv2.LINE_AA)
    for i in range(17):
        cv2.circle(img, tuple(map(int, k[i])), t + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, k[i])), t + 1, color, -1, cv2.LINE_AA)
    return img


def main():
    df = pd.read_csv(BATCH / "summary.csv")
    by_key = {(s["domain"], s["tier"]): s["sample"] for s in select_load_samples(df)}
    cells = [(d, t, by_key[(d, t)]["file_stem"]) for d in DOMAINS for t in TIERS if (d, t) in by_key]
    base, crop, ref = {}, {}, {}
    for _d, _t, stem in cells:
        z = np.load(BATCH / "details" / f"{stem}.npz")
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        base[stem] = mosaic_face(img, z["kpts2d_133"], z["scores_133"])
        ref[stem] = z["kpts2d_133"][:17].astype(float)
        ok = z["scores_133"][:23] >= 0.3
        (x0, y0), (x1, y1) = z["kpts2d_133"][:23][ok].min(0), z["kpts2d_133"][:23][ok].max(0)
        m = 0.25 * max(x1 - x0, y1 - y0)
        crop[stem] = (int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(img.shape[1], x1 + m)), int(min(img.shape[0], y1 + m)))
    results = {}
    for name, _label in MODELS:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            run = build(name, "cuda")
        for _d, _t, stem in cells:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                k, s = run(cv2.imread(str(IMG_DIR / f"{stem}.jpg")))
            k, s = np.asarray(k, float), np.asarray(s, float)
            if k.ndim == 3 and len(k):
                base[stem] = blur_heads(base[stem], k, s)          # every detected head, every model
            results[(name, stem)] = (k, s)
        print(f"{name}: done", flush=True)
    fig, axes = plt.subplots(len(MODELS), len(cells), figsize=(2.0 * len(cells), 2.6 * len(MODELS)))
    for c, (dom, tier, stem) in enumerate(cells):
        x0, y0, x1, y1 = crop[stem]
        axes[0, c].set_title(f"{dom.capitalize()}, {tier}", fontsize=9)
        for r, (name, label) in enumerate(MODELS):
            k, s = results[(name, stem)]
            j = nearest(k, ref[stem]) if k.ndim == 3 else None
            panel = base[stem].copy()
            note = "no detection"
            if j is not None:
                panel = draw(panel, k[j])
                note = f"{len(k)} person(s) detected"
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(panel[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(note, fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
