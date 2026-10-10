"""Compare pose-detection accuracy across three publicly-available
keypoint backbones on the seven `pose_overview.png` example images.

Models compared (all loaded via `rtmlib`):

  * Body           — COCO-17 body-only      (RTMPose, 17 keypoints)
  * BodyWithFeet   — Halpe-style 26 kpts    (body 17 + foot 6 + head/neck/pelvis)
  * Wholebody3d    — RTMW3D 133-kpt 3D      (BIOERA major-25 source)

The script re-uses the sample-selection logic of run_pose_visualization.py
(one high-RULA, median-confidence image from each of the seven Busan
locations) so the comparison is anchored to the exact poses in Figure 2.

Per (model, image) the script reports:

  * mean keypoint confidence over the model's body keypoints
  * %visible: fraction of the model's body keypoints with conf >= 0.5
  * n_detections: number of person bounding boxes (multi-person capability check)

Outputs:

  * model_accuracy_comparison.csv  — full numbers
  * model_accuracy_summary.csv     — per-model means across the 7 images
  * model_accuracy_comparison.png  — grouped bar chart (300 dpi) + .pdf

For the four non-public models (SMPL-X, BioPose-24, HiPART, VEHS-6.7M)
the script writes a separate literature-benchmarks CSV.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
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


log = logging.getLogger("ace_pose.run_model_acc")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


# Per-model body-keypoint indices for fair "%visible" computation.
# Body and BodyWithFeet are top-down RTMPose-family models with their
# own native indexing schemes; the body part starts at 0 in both.
# Wholebody3d follows the COCO-WholeBody 133 layout.

BODY_KPTS = list(range(17))                          # COCO-17 body
BODY_WITH_FEET_KPTS = list(range(23))                # 17 body + 6 feet
WHOLEBODY3D_BODY_KPTS = list(range(17))              # body 17 of 133
WHOLEBODY3D_BIOERA_KPTS = list(range(17)) + list(range(17, 23))  # body + feet


# ----------------------------------------------------------------------
# Sample selection (mirrors run_pose_visualization.py)
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
# Inference wrappers
# ----------------------------------------------------------------------

def _silent(fn, *a, **kw):
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*a, **kw)


class Models:
    """Lazy-init the three rtmlib backbones."""
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
            log.info("Loading rtmlib.BodyWithFeet (Halpe-style 26pt)…")
            cls._bodyfeet = _silent(rtmlib.BodyWithFeet, mode="balanced",
                                    backend="onnxruntime", device="cuda")
        return cls._bodyfeet

    @classmethod
    def wb3d(cls):
        if cls._wb3d is None:
            import rtmlib
            log.info("Loading rtmlib.Wholebody3d (RTMW3D 133pt)…")
            cls._wb3d = _silent(rtmlib.Wholebody3d, mode="balanced",
                                backend="onnxruntime", device="cuda")
        return cls._wb3d


def infer_body(img):
    model = Models.body()
    kpts, scores = _silent(model, img)
    return kpts, scores            # (N, 17, 2), (N, 17)


def infer_bodyfeet(img):
    model = Models.bodyfeet()
    kpts, scores = _silent(model, img)
    return kpts, scores            # (N, K, 2), (N, K)


def infer_wb3d(img):
    model = Models.wb3d()
    out = _silent(model, img)
    # rtmlib.Wholebody3d returns (kpts3d, scores, simcc, kpts2d)
    _kpts3d, scores, _simcc, kpts2d = out
    return kpts2d, scores          # (N, 133, 2) image coordinates, (N, 133)


# ----------------------------------------------------------------------
# Per-image metrics
# ----------------------------------------------------------------------

def primary_idx(centroids: np.ndarray, target: np.ndarray) -> int:
    """Detection whose 2D body centroid is nearest the analysed worker's."""
    return int(np.argmin(np.linalg.norm(centroids - target, axis=1)))


def per_model_metrics(scores: np.ndarray, body_idx: list[int], thresh: float = 0.5):
    body = scores[body_idx]
    return {
        "n_kpts": len(body),
        "mean_conf": float(np.mean(body)),
        "pct_visible": float((body >= thresh).mean()) * 100.0,
        "min_conf": float(np.min(body)),
    }


# ----------------------------------------------------------------------
# Literature accuracy table (non-public models)
# ----------------------------------------------------------------------

LITERATURE = [
    {
        "Model": "SMPL-X body-22",
        "Reference": "Pavlakos et al., CVPR 2019",
        "Benchmark": "EHF: Expressive Hands and Faces",
        "Metric": "Body MPJPE",
        "Value": "≈ 49 mm",
        "Notes": "SMPLify-X iterative fit; ~ minutes per image.",
    },
    {
        "Model": "BioPose-24",
        "Reference": "Cha et al., arXiv 2501.07800 (2025)",
        "Benchmark": "3DPW + biomechanical refinement (OpenSim)",
        "Metric": "Joint-angle MAE (refined)",
        "Value": "≈ 3–5 °",
        "Notes": "Two-stage: monocular + OpenSim; non-real-time.",
    },
    {
        "Model": "HiPART (17 → 48)",
        "Reference": "Zheng et al., CVPR 2025",
        "Benchmark": "Human3.6M (P1)",
        "Metric": "MPJPE",
        "Value": "≈ 37 mm",
        "Notes": "Hierarchical sparse-to-dense decoder; real-time.",
    },
    {
        "Model": "VEHS-6.7M (66 kpts)",
        "Reference": "Wei et al., Ergonomics 2026",
        "Benchmark": "VEHS studio dataset",
        "Metric": "Joint-angle MAE",
        "Value": "≈ 5–7 °",
        "Notes": "WMSD-specific 66-landmark layout; studio capture.",
    },
]


# ----------------------------------------------------------------------
# Bar chart
# ----------------------------------------------------------------------

def render_bar_chart(rows: list[dict], out_png: Path):
    df = pd.DataFrame(rows)
    # Pivot to (image × model) for grouped bars
    metrics = ["mean_conf", "pct_visible"]
    titles = ["Mean keypoint confidence (body)", "% body keypoints with confidence ≥ 0.5"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    images = df["image_label"].unique()
    models = df["model"].unique()
    width = 0.27
    x = np.arange(len(images))
    colors = {"Body (COCO-17)": "#2563eb",
              "BodyWithFeet (Halpe-26)": "#dc2626",
              "Wholebody3d → BIOERA-25": "#16a34a"}
    for ax, met, title in zip(axes, metrics, titles):
        for i, m in enumerate(models):
            sub = df[df["model"] == m].set_index("image_label").loc[images]
            ax.bar(x + (i - 1) * width, sub[met].values, width=width,
                   color=colors.get(m, "#888"), label=m, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(images, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if met == "mean_conf":
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Mean confidence")
        else:
            ax.set_ylim(0.0, 105)
            ax.set_ylabel("% high-confidence keypoints")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Keypoint confidence (not localisation accuracy) on the seven example images",
                 fontsize=13, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str,
                        help="batch directory (e.g. ace_pose/_runs/batch_v1)")
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    out_dir = batch_dir / "analysis"
    fig_dir = out_dir / "figures"

    # 1. Load summary and select samples
    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        p = batch_dir / cand
        if p.exists():
            sum_path = p
            break
    df = pd.read_csv(sum_path)
    df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")
    chosen = select_samples(df)
    log.info("Selected %d sample poses from %s", len(chosen), sum_path.name)

    # 2. Run all three models on each image
    rows = []
    for r in chosen:
        img_path = IMG_DIR / f"{r['file_stem']}.jpg"
        if not img_path.exists():
            img_path = IMG_DIR / f"{r['file_stem']}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("could not read %s", img_path)
            continue
        label = r["location"]
        # the cohort's analysed worker (2D body centroid), so every model is
        # scored on the same person
        target = np.load(batch_dir / "details" / f"{r['file_stem']}.npz")["kpts2d_133"][:17].mean(axis=0)

        # Body (COCO-17)
        t0 = time.time()
        try:
            kpts, scs = infer_body(img)
            idx = primary_idx(kpts[:, :17].mean(axis=1), target) if kpts.shape[0] else None
            if idx is not None:
                m = per_model_metrics(scs[idx], BODY_KPTS)
                m.update({"model": "Body (COCO-17)", "image_label": label,
                          "n_persons": int(kpts.shape[0]),
                          "wall_ms": (time.time() - t0) * 1000})
                rows.append(m)
        except Exception as ex:
            log.warning("Body inference failed on %s: %s", label, ex)

        # BodyWithFeet
        t0 = time.time()
        try:
            kpts, scs = infer_bodyfeet(img)
            idx = primary_idx(kpts[:, :17].mean(axis=1), target) if kpts.shape[0] else None
            if idx is not None:
                k_n = scs.shape[1]
                idx_set = list(range(min(k_n, len(BODY_WITH_FEET_KPTS))))
                m = per_model_metrics(scs[idx], idx_set)
                m.update({"model": "BodyWithFeet (Halpe-26)", "image_label": label,
                          "n_persons": int(kpts.shape[0]),
                          "wall_ms": (time.time() - t0) * 1000})
                rows.append(m)
        except Exception as ex:
            log.warning("BodyWithFeet inference failed on %s: %s", label, ex)

        # Wholebody3d → BIOERA major-25
        t0 = time.time()
        try:
            k3, scs = infer_wb3d(img)
            idx = primary_idx(k3[:, :17].mean(axis=1), target) if k3.shape[0] else None
            if idx is not None:
                m = per_model_metrics(scs[idx], WHOLEBODY3D_BIOERA_KPTS)
                m.update({"model": "Wholebody3d → BIOERA-25", "image_label": label,
                          "n_persons": int(k3.shape[0]),
                          "wall_ms": (time.time() - t0) * 1000})
                rows.append(m)
        except Exception as ex:
            log.warning("Wholebody3d inference failed on %s: %s", label, ex)

        log.info("  %-12s done", label)

    # 3. Write CSVs
    per_image = pd.DataFrame(rows)
    per_image.to_csv(out_dir / "model_accuracy_comparison.csv", index=False)
    summary = per_image.groupby("model").agg(
        n_images=("image_label", "count"),
        mean_conf_mean=("mean_conf", "mean"),
        mean_conf_sd=("mean_conf", "std"),
        pct_visible_mean=("pct_visible", "mean"),
        pct_visible_sd=("pct_visible", "std"),
        wall_ms_mean=("wall_ms", "mean"),
        n_persons_mean=("n_persons", "mean"),
    ).round(2)
    summary.to_csv(out_dir / "model_accuracy_summary.csv")
    log.info("\n%s", summary.to_string())

    # 4. Literature table for non-public models
    lit = pd.DataFrame(LITERATURE)
    lit.to_csv(out_dir / "model_accuracy_literature.csv", index=False)
    log.info("Wrote literature benchmarks (%d rows)", len(lit))

    # 5. Bar chart
    render_bar_chart(rows, fig_dir / "model_accuracy_comparison.png")
    log.info("Wrote %s + .pdf", fig_dir / "model_accuracy_comparison.png")


if __name__ == "__main__":
    main()
