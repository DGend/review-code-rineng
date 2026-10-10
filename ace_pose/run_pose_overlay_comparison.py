"""Side-by-side pose-estimation overlay comparison on the seven
pose_overview example images.

For each of the seven Busan locations we render the SAME image three
times, once per publicly-available rtmlib backbone:

  * Body          — RTMPose-COCO-17 (17 keypoints)
  * BodyWithFeet  — Halpe-style body + feet (23 keypoints)
  * Wholebody3d   — RTMW3D 133-pt, rendered as the BIOERA major-25
                    subset (17 body + 6 feet + derived neck & pelvis)

The face is pixelated using the Wholebody3d face landmarks before any
model's skeleton is drawn, so the privacy mask is identical across
all three model panels of the same worker.

Output: a 3 (rows = models) × 7 (cols = locations) grid PNG/PDF at
`ace_pose/_runs/batch_v1/analysis/figures/pose_overlay_comparison.{png,pdf}`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.analysis import FACTORY_LABEL
from ace_pose.run_pose_visualization import (
    COCO_BODY_EDGES, FOOT_EDGES,
    confidence_color, mosaic_face, crop_around_worker,
)


log = logging.getLogger("ace_pose.run_overlay_cmp")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


def _silent(fn, *a, **kw):
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*a, **kw)


class Models:
    _body = None
    _bodyfeet = None
    _wb3d = None

    @classmethod
    def body(cls):
        if cls._body is None:
            import rtmlib
            log.info("Loading rtmlib.Body (RTMPose-COCO17)…")
            cls._body = _silent(rtmlib.Body, mode="balanced",
                                backend="onnxruntime", device="cuda")
        return cls._body

    @classmethod
    def bodyfeet(cls):
        if cls._bodyfeet is None:
            import rtmlib
            log.info("Loading rtmlib.BodyWithFeet…")
            cls._bodyfeet = _silent(rtmlib.BodyWithFeet, mode="balanced",
                                    backend="onnxruntime", device="cuda")
        return cls._bodyfeet

    @classmethod
    def wb3d(cls):
        if cls._wb3d is None:
            import rtmlib
            log.info("Loading rtmlib.Wholebody3d…")
            cls._wb3d = _silent(rtmlib.Wholebody3d, mode="balanced",
                                backend="onnxruntime", device="cuda")
        return cls._wb3d


# ----------------------------------------------------------------------
# Per-model skeletons
# ----------------------------------------------------------------------

# COCO-17 body edges (re-used)
SKEL_COCO = COCO_BODY_EDGES

# Halpe body+feet — same body edges + 6 foot edges
SKEL_BODYFEET = COCO_BODY_EDGES + FOOT_EDGES

# BIOERA major-25 — body + feet + derived neck/pelvis links
# Indices for BIOERA layout:
#   0..16 = COCO body (RTMW3D output indices 0..16)
#   17..22 = feet (RTMW3D 17..22)
#   23 = derived neck (mid-shoulder = (5 + 6) / 2)
#   24 = derived pelvis (mid-hip = (11 + 12) / 2)
SKEL_BIOERA_EXTRA = [
    (23, 24),                # neck → pelvis
    (5, 23), (6, 23),        # shoulders → neck
    (11, 24), (12, 24),      # hips → pelvis
]


# 2D body centroid of the cohort's analysed worker for the current image, set
# by the driver so every model draws (and masks) the same person.
# ponytail: module-level state, fine for this sequential script.
TARGET_CENTROID: np.ndarray | None = None


def primary_idx(kpts: np.ndarray) -> int:
    if kpts.shape[0] == 0:
        return -1
    if kpts.shape[0] == 1 or TARGET_CENTROID is None:
        return 0
    cents = kpts[:, :17, :2].mean(axis=1)
    return int(np.argmin(np.linalg.norm(cents - TARGET_CENTROID, axis=1)))


def draw_keypoints(img_bgr, kpts2d, scores, edges, indices,
                   body_radius=7, edge_thick=5, conf_thresh=0.2):
    """Draw the skeleton edges and keypoint dots for a subset of joints."""
    for s, t in edges:
        if s >= scores.shape[0] or t >= scores.shape[0]:
            continue
        if scores[s] < conf_thresh or scores[t] < conf_thresh:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        c = 0.5 * (scores[s] + scores[t])
        cv2.line(img_bgr, p0, p1, confidence_color(c), edge_thick,
                 lineType=cv2.LINE_AA)
    for j in indices:
        if j >= scores.shape[0]:
            continue
        if scores[j] < conf_thresh:
            continue
        p = tuple(map(int, kpts2d[j]))
        cv2.circle(img_bgr, p, body_radius, confidence_color(scores[j]),
                   -1, lineType=cv2.LINE_AA)
        cv2.circle(img_bgr, p, body_radius, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)
    return img_bgr


def render_body(img: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run Body and draw COCO-17 skeleton. Returns (rendered, mean_conf, n_visible)."""
    out = img.copy()
    try:
        kpts, scores = _silent(Models.body(), out)
    except Exception as ex:
        log.warning("Body inference failed: %s", ex)
        return out, 0.0, 0
    if kpts.shape[0] == 0:
        return out, 0.0, 0
    idx = primary_idx(kpts)
    k, s = kpts[idx], scores[idx]
    out = draw_keypoints(out, k, s, SKEL_COCO, list(range(17)))
    return out, float(s[:17].mean()), int((s[:17] >= 0.5).sum())


def render_bodyfeet(img: np.ndarray) -> tuple[np.ndarray, float, int]:
    out = img.copy()
    try:
        kpts, scores = _silent(Models.bodyfeet(), out)
    except Exception as ex:
        log.warning("BodyWithFeet inference failed: %s", ex)
        return out, 0.0, 0
    if kpts.shape[0] == 0:
        return out, 0.0, 0
    idx = primary_idx(kpts)
    k, s = kpts[idx], scores[idx]
    n = min(s.shape[0], 23)
    out = draw_keypoints(out, k, s, SKEL_BODYFEET, list(range(n)))
    return out, float(s[:n].mean()), int((s[:n] >= 0.5).sum())


def render_bioera(img: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run Wholebody3d (133-pt) and draw BIOERA major-25 (body + feet +
    derived neck & pelvis)."""
    out = img.copy()
    try:
        k3, scores, _simcc, kpts2d_133 = _silent(Models.wb3d(), out)
    except Exception as ex:
        log.warning("Wholebody3d inference failed: %s", ex)
        return out, 0.0, 0
    if kpts2d_133.shape[0] == 0:
        return out, 0.0, 0
    idx = primary_idx(kpts2d_133)
    k, s = kpts2d_133[idx], scores[idx]

    # Append derived neck (mid-shoulder) and pelvis (mid-hip) at indices 23, 24
    neck_xy = 0.5 * (k[5] + k[6])
    pelv_xy = 0.5 * (k[11] + k[12])
    k_ext = np.vstack([k[:23], neck_xy, pelv_xy])           # (25, 2)
    s_ext = np.concatenate([s[:23], [min(s[5], s[6])], [min(s[11], s[12])]])

    out = draw_keypoints(out, k_ext, s_ext,
                         SKEL_BODYFEET + SKEL_BIOERA_EXTRA,
                         list(range(25)))
    return out, float(s_ext.mean()), int((s_ext >= 0.5).sum())


# ----------------------------------------------------------------------
# Sample selection (same logic as run_pose_visualization.py)
# ----------------------------------------------------------------------

def select_samples(df: pd.DataFrame) -> list[pd.Series]:
    samples = []
    for loc in sorted(df["location"].dropna().unique()):
        sub = df.loc[(df["location"] == loc) & (df["status"] == "ok")
                     & (df["kpt_conf_mean"] > 0.55)]
        if sub.empty:
            continue
        sub = sub.sort_values("rula_grand", ascending=False)
        top_q = sub.head(max(50, len(sub) // 4))
        top_q = top_q.sort_values("kpt_conf_mean")
        chosen = top_q.iloc[len(top_q) // 2]
        samples.append(chosen)
        if len(samples) >= 7:
            break
    return samples


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    fig_dir = batch_dir / "analysis" / "figures"

    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        p = batch_dir / cand
        if p.exists():
            sum_path = p
            break
    df = pd.read_csv(sum_path)
    df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")
    chosen = select_samples(df)
    log.info("Selected %d sample poses", len(chosen))

    # Pre-load all three models once
    Models.body()
    Models.bodyfeet()
    Models.wb3d()

    # Pre-compute the mosaicked image and crop bounds from RTMW3D for
    # each example, so all 3 panels per column share the same crop.
    columns = []
    for r in chosen:
        stem = r["file_stem"]
        img_path = IMG_DIR / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        global TARGET_CENTROID
        TARGET_CENTROID = np.load(batch_dir / "details" / f"{stem}.npz")["kpts2d_133"][:17].mean(axis=0)
        # Run WB3D once to determine face mosaic + crop window
        k3, scores, _simcc, kpts2d_133 = _silent(Models.wb3d(), img)
        if kpts2d_133.shape[0] == 0:
            log.warning("no WB3D detection on %s", stem)
            continue
        idx = primary_idx(kpts2d_133)
        kp, sc = kpts2d_133[idx], scores[idx]
        img_mos = mosaic_face(img.copy(), kp, sc)

        # Determine common crop bounds using 91 points (body+feet+face)
        # (face mosaicked but face landmarks still inform the crop bbox)
        pts = kp[:91]
        cs = sc[:91]
        valid = cs > 0.15
        if valid.sum() < 4:
            x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
        else:
            pts = pts[valid]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            pad_x = (x_max - x_min) * 0.12
            pad_y = (y_max - y_min) * 0.12
            x0 = int(max(0, x_min - pad_x))
            x1 = int(min(img.shape[1], x_max + pad_x))
            y0 = int(max(0, y_min - pad_y))
            y1 = int(min(img.shape[0], y_max + pad_y))

        # Render the three panels on the (uncropped) mosaicked image
        body_render, body_mc, body_v = render_body(img_mos)
        bf_render, bf_mc, bf_v = render_bodyfeet(img_mos)
        bio_render, bio_mc, bio_v = render_bioera(img_mos)

        # crop each
        def _crop(im):
            return im[y0:y1, x0:x1]
        columns.append({
            "label": r["location"],
            "rula": r.get("rula_grand", float("nan")),
            "panels": [
                ("Body (COCO-17)",          _crop(body_render), body_mc, body_v, 17),
                ("BodyWithFeet (Halpe-26)", _crop(bf_render),   bf_mc,   bf_v,   23),
                ("Wholebody3d → BIOERA-25", _crop(bio_render),  bio_mc,  bio_v,  25),
            ],
        })
        log.info("  %-14s rendered", r["location"])

    if not columns:
        log.error("no columns rendered")
        return

    # ------ Layout: 3 rows × 7 cols ------
    nc = len(columns)
    nr = 3
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 3.0 * nr))
    axes = np.atleast_2d(axes)
    model_names = ["Body (COCO-17)",
                   "BodyWithFeet (Halpe-26)",
                   "Wholebody3d → BIOERA-25"]
    for ci, col in enumerate(columns):
        for ri in range(3):
            ax = axes[ri, ci]
            name, im, mc, v, kn = col["panels"][ri]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col["label"], fontsize=10, weight="bold", pad=4)
            # bottom strip text
            cap = f"n={kn}  mean={mc:.2f}  vis={v}/{kn}"
            ax.set_xlabel(cap, fontsize=8.5, labelpad=2)
        # row labels on the leftmost column
    for ri, mname in enumerate(model_names):
        axes[ri, 0].text(-0.07, 0.5, mname, transform=axes[ri, 0].transAxes,
                         ha="right", va="center", rotation=90,
                         fontsize=11, weight="bold")

    fig.suptitle(
        "Pose-estimation overlay comparison: 3 publicly-available backbones × "
        "7 Busan locations  (face mosaicked from RTMW3D 68-pt face landmarks)",
        fontsize=12, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out_png = fig_dir / "pose_overlay_comparison.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s + .pdf", out_png)


if __name__ == "__main__":
    main()
