"""Render worker-pose visualization panels for the manuscript.

For each selected sample image we produce a privacy-respecting
visualization that pairs the raw imagery with the BIOERA outputs:

  - the worker's face is **pixelated** using the COCO-WholeBody face
    landmarks (indices 23..90) so subjects are not identifiable;
  - the RTMW3D 133-keypoint skeleton is overlaid; joint markers are
    coloured by per-keypoint confidence, edge thickness by edge weight;
  - the figure caption strip carries the location, ground-truth and
    predicted process labels, and the OWAS / RULA / REBA scores
    produced by the BIOERA pipeline.

Output: a multi-panel PNG (300 dpi) + matching PDF in
`batch/analysis/figures/pose_overview.{png,pdf}`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.analysis import FACTORY_LABEL
from ace_pose.batch_infer import PROCESS_LABEL
from ace_pose.run_plots import PROCESS_LABEL_EN, _romanise_process
from ace_pose.spine_overlay import (
    kp25_px_from_133, silhouette_spine_cached, draw_spine_curve_bgr,
    attach_node_load,
)


log = logging.getLogger("ace_pose.run_pose_vis")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"

# COCO-WholeBody body skeleton (17pt subset)
COCO_BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                   # face → ears
    (5, 6),                                           # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),                  # arms
    (5, 11), (6, 12), (11, 12),                       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),           # legs
]

# Foot keypoints (COCO-WholeBody indices 17..22)
FOOT_EDGES = [
    (15, 17), (15, 18), (15, 19),                     # L ankle → big-toe/small-toe/heel
    (16, 20), (16, 21), (16, 22),                     # R ankle → big-toe/small-toe/heel
]

# Face skeleton (68-point landmark model, COCO indices 23..90).
# We connect the canonical face contours so a reviewer can see we are
# using whole-body SOTA detection, not just the body-17 subset.
def _face_edges_offset(o: int = 23):
    """Return the canonical 68-pt face contour edges with COCO offset."""
    chains = [
        list(range(0, 17)),    # jawline
        list(range(17, 22)),   # left eyebrow
        list(range(22, 27)),   # right eyebrow
        list(range(27, 31)),   # nose bridge
        list(range(31, 36)),   # nose lower
        list(range(36, 42)) + [36],          # left eye (closed loop)
        list(range(42, 48)) + [42],          # right eye (closed loop)
        list(range(48, 60)) + [48],          # outer mouth (closed loop)
        list(range(60, 68)) + [60],          # inner mouth (closed loop)
    ]
    edges = []
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append((o + a, o + b))
    return edges


FACE_EDGES = _face_edges_offset(23)

# Hand keypoints (91..132) are EXCLUDED from BIOERA. We omit hand
# skeleton edges entirely so the figure reflects the production layout.

# Combined set of indices we DRAW: 17 body + 6 feet + 68 face = 91 keypoints
DRAWN_INDICES = list(range(0, 91))


def mosaic_face(image_bgr: np.ndarray, kpts2d: np.ndarray,
                scores: np.ndarray, pad: float = 1.6,
                block: int = 12) -> np.ndarray:
    """In-place pixelate a square region covering the head: the confident face
    landmarks (COCO-WholeBody 23..90) together with the nose, eyes, and ears,
    enlarged by `pad`, and never smaller than half the shoulder width. Face
    landmarks alone can sit off the face in profile views, so the head body
    keypoints anchor the region."""
    ids = list(range(23, 91)) + [0, 1, 2, 3, 4]
    pts = kpts2d[ids][scores[ids] > 0.10]
    if len(pts) < 3:
        return image_bgr
    h, w = image_bgr.shape[:2]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    shoulder = float(np.linalg.norm(kpts2d[5] - kpts2d[6])) if min(scores[5], scores[6]) > 0.10 else 0.0
    half = max(x_max - x_min, y_max - y_min, 0.5 * shoulder) * pad / 2
    x0 = int(max(0, cx - half))
    x1 = int(min(w, cx + half))
    y0 = int(max(0, cy - half))
    y1 = int(min(h, cy + half))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return image_bgr

    region = image_bgr[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    nbx = max(1, rw // block)
    nby = max(1, rh // block)
    pix = cv2.resize(region, (nbx, nby), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(pix, (rw, rh), interpolation=cv2.INTER_NEAREST)
    image_bgr[y0:y1, x0:x1] = pix
    return image_bgr


def confidence_color(c: float) -> tuple[int, int, int]:
    """Confidence → BGR colour (red low → green high)."""
    c = float(np.clip(c, 0, 1))
    r = int(255 * (1.0 - c))
    g = int(255 * c)
    return (0, g, r)            # BGR


def draw_skeleton(image_bgr: np.ndarray, kpts2d: np.ndarray,
                  scores: np.ndarray,
                  conf_threshold: float = 0.20,
                  body_thick: int = 6, foot_thick: int = 3,
                  face_thick: int = 2,
                  body_radius: int = 7, foot_radius: int = 4,
                  face_radius: int = 3) -> np.ndarray:
    """Overlay all 91 RTMW3D keypoints we USE in BIOERA:
       17 body + 6 feet + 68 face = 91 (hand keypoints 91–132 excluded).
    """
    # ---- Face contour skeleton (68 landmarks) ----
    for s, t in FACE_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        cv2.line(image_bgr, p0, p1, (255, 220, 80), face_thick,
                 lineType=cv2.LINE_AA)

    # ---- Foot skeleton (6 keypoints) ----
    for s, t in FOOT_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        c_avg = 0.5 * (scores[s] + scores[t])
        col = confidence_color(c_avg)
        cv2.line(image_bgr, p0, p1, col, foot_thick, lineType=cv2.LINE_AA)

    # ---- Body skeleton (17 keypoints) ----
    for s, t in COCO_BODY_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        c_avg = 0.5 * (scores[s] + scores[t])
        col = confidence_color(c_avg)
        cv2.line(image_bgr, p0, p1, col, body_thick, lineType=cv2.LINE_AA)

    # ---- Markers ----
    # Face landmarks: small dots in yellow on a black halo for visibility
    # over the pixelated face region.
    for j in range(23, 91):
        if scores[j] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[j]))
        cv2.circle(image_bgr, p, face_radius + 1, (0, 0, 0), -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, face_radius, (80, 220, 255), -1,
                   lineType=cv2.LINE_AA)

    # Foot keypoints
    for j in range(17, 23):
        if scores[j] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[j]))
        col = confidence_color(scores[j])
        cv2.circle(image_bgr, p, foot_radius, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, foot_radius, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)

    # Body keypoints (largest, on top)
    for j in range(17):
        if scores[j] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[j]))
        col = confidence_color(scores[j])
        cv2.circle(image_bgr, p, body_radius, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, body_radius, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)
    return image_bgr


def crop_around_worker(image_bgr: np.ndarray, kpts2d: np.ndarray,
                       scores: np.ndarray, pad: float = 0.12) -> np.ndarray:
    """Tight crop around the worker using all 91 displayed keypoints
    (body 17 + feet 6 + face 68)."""
    pts = kpts2d[:91]
    cs = scores[:91]
    valid = cs > 0.15
    if valid.sum() < 4:
        return image_bgr
    pts = pts[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    h, w = image_bgr.shape[:2]
    pad_x = (x_max - x_min) * pad
    pad_y = (y_max - y_min) * pad
    x0 = int(max(0, x_min - pad_x))
    x1 = int(min(w, x_max + pad_x))
    y0 = int(max(0, y_min - pad_y))
    y1 = int(min(h, y_max + pad_y))
    return image_bgr[y0:y1, x0:x1]


def make_panel(row: pd.Series, batch_dir: Path) -> np.ndarray:
    """One panel: image + face-mosaicked skeleton overlay + caption strip."""
    stem = row["file_stem"]
    npz = np.load(batch_dir / "details" / f"{stem}.npz")
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    kpts2d = npz["kpts2d_133"]
    scores = npz["scores_133"]
    img = mosaic_face(img, kpts2d, scores)
    img = draw_skeleton(img, kpts2d, scores)
    # Silhouette spine (DSE v2) overlay — cached per worker
    kp25_px, cf25_px = kp25_px_from_133(kpts2d, scores)
    spine = silhouette_spine_cached(
        img, kp25_px, cf25_px,
        batch_dir / "analysis" / "_spine_cache", stem,
    )
    spine = attach_node_load(spine, npz["kpts_m25"], npz["conf_m25"])
    img = draw_spine_curve_bgr(img, spine)
    # rebuild keypoints in the crop frame
    img_crop = crop_around_worker(img, kpts2d, scores)
    # convert BGR -> RGB for matplotlib
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    return img_crop


def select_samples(df: pd.DataFrame, n_each: int = 1) -> list[pd.Series]:
    """Pick a representative high-risk pose from each location."""
    samples = []
    for loc in sorted(df["location"].dropna().unique()):
        sub = df.loc[(df["location"] == loc) & (df["status"] == "ok")
                     & (df["kpt_conf_mean"] > 0.55)]
        if sub.empty:
            continue
        # Sort by RULA grand desc, pick the median-confidence row from the top quartile
        sub = sub.sort_values("rula_grand", ascending=False)
        top_q = sub.head(max(50, len(sub) // 4))
        # Among the top quartile, pick the row with median pose-confidence
        top_q = top_q.sort_values("kpt_conf_mean")
        chosen = top_q.iloc[len(top_q) // 2]
        samples.append(chosen)
        if len(samples) >= 7:
            break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--n_each", type=int, default=1)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    fig_dir = batch_dir / "analysis" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # Prefer the most-enriched summary
    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        path = batch_dir / cand
        if path.exists():
            sum_path = path
            break
    df = pd.read_csv(sum_path)
    df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")
    log.info("Loaded %d rows from %s", len(df), sum_path)

    chosen = select_samples(df, n_each=args.n_each)
    log.info("Selected %d sample poses", len(chosen))

    n = len(chosen)
    cols = 4 if n > 4 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 5.0 * rows))
    axes = np.atleast_1d(axes).flatten()
    plt.rcParams["axes.titlesize"] = 9

    for ax, row in zip(axes, chosen):
        img = make_panel(row, batch_dir)
        ax.imshow(img)
        ax.axis("off")
        # Build caption
        proc_kor = row.get("process_label", "?")
        proc_en = _romanise_process(proc_kor)
        pred_id = row.get("pred_process_id", float("nan"))
        pred_en = _romanise_process(int(pred_id)) if not pd.isna(pred_id) else "-"
        conf = row.get("kpt_conf_mean", float("nan"))
        owas = int(row["owas_action"])
        rula = int(row["rula_grand"])
        reba = int(row["reba_grand"])
        biomech = row.get("biomech_L5S1_Nm", float("nan"))
        title = (
            f"{row['location']}  ·  GT: {proc_en}  ·  Pred: {pred_en}\n"
            f"OWAS={owas}  RULA={rula}  REBA={reba}"
            + (f"  ·  L5/S1={biomech:.0f} Nm" if not pd.isna(biomech) else "")
            + f"  ·  conf={conf:.2f}"
        )
        ax.set_title(title, fontsize=9, pad=4)

    # blank out unused axes
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(
        "Worker pose visualisation (face mosaicked): RTMW3D 91-kpt overlay "
        "(body 17 + feet 6 + face 68; hand keypoints 91–132 excluded per BIOERA) "
        "+ BIOERA scores",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = fig_dir / "pose_overview.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Wrote %s + .pdf", out)


if __name__ == "__main__":
    main()
