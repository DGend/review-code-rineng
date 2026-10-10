"""Render the 9 (models) × 9 (Busan environment×risk-tier images)
comparison required for Section 3.0.1 of the BIOERA paper.

Outputs:

  * For each of the 9 models, a 3×3 (domains × risk tiers) figure:
        `model_grid_<short>.png` + `.pdf`
  * A unified 9 × 9 mega-grid:
        `model_mega_grid.png` + `.pdf`

Three of the nine models are real ONNX inference (Body, BodyWithFeet,
Wholebody3d). The remaining six are *schematic* — their joint
convention is rendered at anatomically-derived positions from the
RTMW3D 133-pt detection, *not* from a native pretrained model. Each
panel carries a REAL or SCHEMATIC badge so this distinction is
unambiguous.
"""
from __future__ import annotations

import argparse
import logging
import sys
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
from ace_pose.model_schemas import MODELS_9
from ace_pose.run_pose_visualization import (
    confidence_color, mosaic_face,
)
from ace_pose.run_env_load_visualization import select_load_samples


log = logging.getLogger("ace_pose.run_full_model_cmp")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


def draw_skeleton(img, kpts, scores, edges,
                  body_radius=6, edge_thick=4, conf_thresh=0.2):
    M = scores.shape[0]
    for s, t in edges:
        if s >= M or t >= M:
            continue
        if scores[s] < conf_thresh or scores[t] < conf_thresh:
            continue
        p0 = tuple(map(int, kpts[s]))
        p1 = tuple(map(int, kpts[t]))
        cv2.line(img, p0, p1, confidence_color(0.5 * (scores[s] + scores[t])),
                 edge_thick, lineType=cv2.LINE_AA)
    for j in range(M):
        if scores[j] < conf_thresh:
            continue
        p = tuple(map(int, kpts[j]))
        cv2.circle(img, p, body_radius, confidence_color(scores[j]), -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(img, p, body_radius, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)
    return img


def common_crop(k133: np.ndarray, s133: np.ndarray, img_shape):
    """Crop box using the 91 displayed RTMW3D landmarks."""
    pts = k133[:91]
    cs = s133[:91]
    valid = cs > 0.15
    if valid.sum() < 4:
        return 0, 0, img_shape[1], img_shape[0]
    pts = pts[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 0.12
    pad_x = (x_max - x_min) * pad
    pad_y = (y_max - y_min) * pad
    x0 = int(max(0, x_min - pad_x))
    x1 = int(min(img_shape[1], x_max + pad_x))
    y0 = int(max(0, y_min - pad_y))
    y1 = int(min(img_shape[0], y_max + pad_y))
    return x0, y0, x1, y1


def render_one_panel(ax, img_rgb, model_name, is_real, label,
                     n_kpts, mean_conf, vis_count):
    ax.imshow(img_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    badge = "REAL" if is_real else "SCHEMATIC"
    badge_color = "#16a34a" if is_real else "#dc2626"
    title = f"{model_name}\n[{label}]"
    ax.set_title(title, fontsize=8.0, weight="bold", pad=3)
    # bottom strip: n=… mean=… vis=…
    cap = f"n={n_kpts}  mean={mean_conf:.2f}  vis={vis_count}/{n_kpts}"
    ax.set_xlabel(cap, fontsize=7.0, labelpad=2)
    # badge in upper-left corner of axes
    ax.text(0.02, 0.97, badge, transform=ax.transAxes,
            fontsize=7.5, weight="bold", color="white",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc=badge_color, ec="none"))


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
    samples = select_load_samples(df)
    log.info("Selected %d (domain, tier) samples", len(samples))

    domains = ["manufacturing", "eldercare", "logistics"]
    tiers = ["low", "medium", "high"]
    by_key = {(s["domain"], s["tier"]): s for s in samples}

    # For each (domain, tier), pre-compute the mosaicked image + crop bounds.
    # We run RTMW3D once for face mosaicking + crop and re-use for schematic
    # mappings.
    from ace_pose import model_schemas
    from ace_pose.model_schemas import _wb3d, _silent
    image_cache: dict[tuple[str, str], dict] = {}
    for dom in domains:
        for ti in tiers:
            entry = by_key.get((dom, ti))
            if entry is None:
                continue
            row = entry["sample"]
            stem = row["file_stem"]
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
            if img is None:
                continue
            try:
                _k3, _s, _simcc, k133 = _silent(_wb3d(), img)
            except Exception as ex:
                log.warning("WB3D failed on %s: %s", stem, ex)
                continue
            if k133.shape[0] == 0:
                continue
            target = np.load(batch_dir / "details" / f"{stem}.npz")["kpts2d_133"][:17].mean(axis=0)
            idx = int(np.argmin(np.linalg.norm(k133[:, :17].mean(axis=1) - target, axis=1)))
            kp = k133[idx]; sc = _s[idx]
            img_mos = mosaic_face(img.copy(), kp, sc)
            x0, y0, x1, y1 = common_crop(kp, sc, img.shape)
            image_cache[(dom, ti)] = {
                "img_mos": img_mos,
                "crop": (x0, y0, x1, y1),
                "label": f"{dom.title()} • {ti.upper()}",
                "loc": row["location"],
                "target": target,
            }
            log.info("Cached %s/%s = %s", dom, ti, row["location"])

    # ------------------------------------------------------------------
    # Generate per-model 3×3 grids
    # ------------------------------------------------------------------
    per_model_panels: dict[str, list[dict]] = {}
    for mspec in MODELS_9:
        log.info("Rendering model: %s (real=%s)", mspec.name, mspec.is_real)
        per_model_panels[mspec.short] = []
        fig, axes = plt.subplots(3, 3, figsize=(11, 12))
        for di, dom in enumerate(domains):
            for ti, tier in enumerate(tiers):
                ax = axes[di, ti]
                entry = image_cache.get((dom, tier))
                if entry is None:
                    ax.set_visible(False); continue
                img = entry["img_mos"].copy()
                x0, y0, x1, y1 = entry["crop"]
                model_schemas.TARGET_CENTROID = entry["target"]
                res = mspec.infer(img)
                if res is None:
                    ax.text(0.5, 0.5, "no detection",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=9, color="#aaa")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                img_drawn = draw_skeleton(img, res["kpts"], res["scores"],
                                          res["edges"])
                cropped = img_drawn[y0:y1, x0:x1]
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                nk = res["kpts"].shape[0]
                mc = float(res["scores"].mean())
                vc = int((res["scores"] >= 0.5).sum())
                render_one_panel(
                    ax, rgb, mspec.name, mspec.is_real,
                    f"{entry['label']}",
                    nk, mc, vc,
                )
                per_model_panels[mspec.short].append({
                    "domain": dom, "tier": tier, "label": entry["loc"],
                    "panel_rgb": rgb,
                    "n_kpts": nk, "mean_conf": mc, "vis": vc,
                })
        fig.suptitle(
            f"{mspec.name}: pose prediction on 9 Busan domain × risk-tier images "
            f"({'REAL inference' if mspec.is_real else 'SCHEMATIC (RTMW3D-derived)'})",
            fontsize=12, weight="bold", y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = fig_dir / f"model_grid_{mspec.short}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        log.info("  → %s", out_png.name)

    # ------------------------------------------------------------------
    # Mega 9×9 grid: rows = models, cols = images
    # ------------------------------------------------------------------
    log.info("Rendering 9×9 mega-grid…")
    n_models = len(MODELS_9)
    n_imgs = len(image_cache)
    fig, axes = plt.subplots(n_models, n_imgs, figsize=(2.2 * n_imgs,
                                                         2.6 * n_models))
    col_labels = []
    col_keys = []
    for dom in domains:
        for tier in tiers:
            if (dom, tier) in image_cache:
                col_keys.append((dom, tier))
                col_labels.append(f"{dom[:4].title()} • {tier.upper()}")
    for ri, mspec in enumerate(MODELS_9):
        for ci, key in enumerate(col_keys):
            ax = axes[ri, ci]
            entry = image_cache[key]
            img = entry["img_mos"].copy()
            x0, y0, x1, y1 = entry["crop"]
            res = mspec.infer(img)
            if res is None:
                ax.text(0.5, 0.5, "no det.", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#aaa")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                img_drawn = draw_skeleton(img, res["kpts"], res["scores"],
                                          res["edges"], body_radius=4,
                                          edge_thick=3)
                cropped = img_drawn[y0:y1, x0:x1]
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                ax.set_xticks([]); ax.set_yticks([])
                nk = res["kpts"].shape[0]
                mc = float(res["scores"].mean())
                ax.set_xlabel(f"n={nk} m={mc:.2f}", fontsize=6.5, labelpad=1)
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=8.5, weight="bold")
        # row label
        badge = "R" if mspec.is_real else "S"
        bc = "#16a34a" if mspec.is_real else "#dc2626"
        axes[ri, 0].text(-0.18, 0.5, f"{mspec.name}",
                         transform=axes[ri, 0].transAxes,
                         ha="right", va="center", rotation=0,
                         fontsize=8.5, weight="bold")
        axes[ri, 0].text(-0.02, 0.5, badge,
                         transform=axes[ri, 0].transAxes,
                         ha="right", va="center", color="white",
                         fontsize=7.5, weight="bold",
                         bbox=dict(boxstyle="round,pad=0.18", fc=bc, ec="none"))
    fig.suptitle(
        "Pose prediction comparison: 9 keypoint conventions × 9 Busan "
        "environment × risk-tier images",
        fontsize=13, weight="bold", y=0.995,
    )
    # ---- Badge legend (explains R / S badges shown next to each row) ----
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    badge_handles = [
        Line2D([], [], marker="s", linestyle="none",
               markerfacecolor="#16a34a", markeredgecolor="none",
               markersize=14,
               label="R :  REAL inference (native pretrained model weights "
                     "applied to the image)"),
        Line2D([], [], marker="s", linestyle="none",
               markerfacecolor="#dc2626", markeredgecolor="none",
               markersize=14,
               label="S :  SCHEMATIC (no public weights available; the "
                     "convention's joints are placed at anatomically-derived "
                     "positions inferred from the RTMW3D 133-pt detection: "
                     "shows the joint LAYOUT, not the model's actual accuracy)"),
    ]
    fig.legend(handles=badge_handles, loc="upper center",
               ncols=1, bbox_to_anchor=(0.5, 0.978),
               frameon=True, fontsize=9.5,
               title="Row badges next to each model name "
                     "(also shown per-panel in the per-model figures):",
               title_fontsize=10)
    # Per-panel sub-caption note
    fig.text(0.5, 0.945,
             "Per-panel sub-caption  '·  n = joint count, m = mean keypoint "
             "confidence (over the model's body keypoints) ·",
             ha="center", va="top", fontsize=9, style="italic", color="#444")
    fig.tight_layout(rect=(0.04, 0, 1, 0.93))
    out_png = fig_dir / "model_mega_grid.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s + .pdf", out_png)


if __name__ == "__main__":
    main()
