"""Per-environment worker-load visualization.

Selects a low-, medium-, and high-risk sample from each of the three
work environments (Manufacturing / Eldercare / Logistics), reruns the
91-keypoint RTMW3D overlay, computes V (Shapley) per-joint
attribution, and renders a 3x3 figure where:

  - Each row is one work environment.
  - Each column is one REBA-action-level tier (low / medium / high).
  - The face is mosaicked for privacy.
  - The 91 RTMW3D keypoints we KEEP (body 17 + feet 6 + face 68;
    hand 91..132 dropped) are overlaid with skeleton edges.
  - The body joints are RE-SIZED and RE-COLOURED to encode each joint's
    Shapley contribution to the REBA grand score, giving a visual
    "load heatmap" directly on the worker.
  - A coloured badge on each panel shows the OWAS, RULA, and REBA
    action levels and the L5/S1 static torque.

The output is `pose_load_overview.{png,pdf}` (300 dpi PNG + vector
PDF).
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
from ace_pose.ergo_v2.context_policy import cohort_context  # posture only, see context_policy
from ace_pose.ergo_v2 import score_all
from ace_pose.major_joints import (
    Major25Pose, MAJOR25_NAMES, coco_wholebody_to_major25,
)
from ace_pose.run_plots import PROCESS_LABEL_EN, _romanise_process
from ace_pose.run_pose_visualization import (
    COCO_BODY_EDGES, DRAWN_INDICES, FACE_EDGES, FOOT_EDGES,
    confidence_color, crop_around_worker, mosaic_face,
)
from ace_pose.spine_overlay import (
    kp25_px_from_133, silhouette_spine_cached, draw_spine_curve_bgr,
    attach_node_load,
)
from ace_pose.xai import attribute_reba


log = logging.getLogger("ace_pose.run_env_load_vis")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


ACTION_COLOR_HEX = {
    0: "#16a34a", 1: "#22c55e",   # low
    2: "#eab308",                 # medium
    3: "#f97316",                 # high
    4: "#dc2626",                 # very high
}
ACTION_LABEL = {
    0: "Negligible",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very high",
}


# Major-25 body joint → COCO body indices in the 91-pt overlay used for drawing
MAJOR25_TO_COCO_BODY = {
    "SHOULDER_LEFT": 5, "SHOULDER_RIGHT": 6,
    "ELBOW_LEFT": 7, "ELBOW_RIGHT": 8,
    "WRIST_LEFT": 9, "WRIST_RIGHT": 10,
    "HIP_LEFT": 11, "HIP_RIGHT": 12,
    "KNEE_LEFT": 13, "KNEE_RIGHT": 14,
    "ANKLE_LEFT": 15, "ANKLE_RIGHT": 16,
    # NECK/PELVIS are derived from neighbouring COCO points (no fixed index)
}


def select_load_samples(df: pd.DataFrame) -> list[dict]:
    """Pick three samples per domain spanning low/medium/high REBA action."""
    out = []
    for dom in ("manufacturing", "eldercare", "logistics"):
        sub = df.loc[
            (df["domain"] == dom)
            & (df["status"] == "ok")
            & (df["kpt_conf_mean"] > 0.50)
        ].copy()
        if sub.empty:
            continue
        for tier, action_levels in [
            ("low", (1,)),
            ("medium", (2,)),
            ("high", (3, 4)),
        ]:
            cand = sub.loc[sub["reba_action"].isin(action_levels)]
            if cand.empty:
                cand = sub.loc[sub["reba_action"] == max(sub["reba_action"])]
            cand = cand.sort_values("kpt_conf_mean", ascending=False)
            chosen = cand.iloc[len(cand) // 4]
            out.append({"sample": chosen, "domain": dom, "tier": tier})
    return out


def joint_shapley_for_reba(row: pd.Series, batch_dir: Path
                           ) -> dict[str, float] | None:
    """Recompute REBA + Shapley attribution from the stored detail npz."""
    stem = row["file_stem"]
    npz_path = batch_dir / "details" / f"{stem}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    kpts2d = data["kpts2d_133"]
    scores = data["scores_133"]
    kpts3d = data["kpts3d_133"]
    m25 = coco_wholebody_to_major25(kpts3d, scores)
    result = score_all(m25,
                       world_up=np.array([0.0, -1.0, 0.0]),
                       walking=False, **cohort_context(row.get("process_label")),
                       domain=row.get("domain"),
                       process_id=int(row.get("process_id", 0)) or None,
                       bilateral=False)
    attribution = attribute_reba(result.reba)
    return attribution.joint_shapley


def draw_pose_with_load(image_bgr: np.ndarray,
                        kpts2d: np.ndarray, scores: np.ndarray,
                        joint_shapley: dict[str, float] | None,
                        conf_threshold: float = 0.20) -> np.ndarray:
    """Skeleton + face landmarks + per-joint load-coloured markers."""
    # Face contour lines (small)
    for s, t in FACE_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        cv2.line(image_bgr, p0, p1, (255, 220, 80), 2, lineType=cv2.LINE_AA)
    # Foot lines
    for s, t in FOOT_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        col = confidence_color(0.5 * (scores[s] + scores[t]))
        cv2.line(image_bgr, p0, p1, col, 3, lineType=cv2.LINE_AA)
    # Body skeleton
    for s, t in COCO_BODY_EDGES:
        if scores[s] < conf_threshold or scores[t] < conf_threshold:
            continue
        p0 = tuple(map(int, kpts2d[s]))
        p1 = tuple(map(int, kpts2d[t]))
        col = confidence_color(0.5 * (scores[s] + scores[t]))
        cv2.line(image_bgr, p0, p1, col, 6, lineType=cv2.LINE_AA)

    # Face markers (small dots)
    for j in range(23, 91):
        if scores[j] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[j]))
        cv2.circle(image_bgr, p, 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, 3, (80, 220, 255), -1, lineType=cv2.LINE_AA)
    # Foot markers
    for j in range(17, 23):
        if scores[j] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[j]))
        col = confidence_color(scores[j])
        cv2.circle(image_bgr, p, 5, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Body markers: SIZE and COLOUR encode Shapley load contribution.
    # Default if no Shapley info: pose-confidence colour.
    max_shap = (max(joint_shapley.values()) if joint_shapley
                else 1.0)
    max_shap = max(max_shap, 1.0)
    for name, coco_idx in MAJOR25_TO_COCO_BODY.items():
        if scores[coco_idx] < conf_threshold:
            continue
        p = tuple(map(int, kpts2d[coco_idx]))
        if joint_shapley is not None:
            v = joint_shapley.get(name, 0.0) / max_shap   # 0..1
            v = float(np.clip(v, 0.0, 1.0))
            # Red intensity scales with load contribution
            radius = int(8 + 14 * v)
            # halo
            cv2.circle(image_bgr, p, radius + 4, (40, 40, 40), -1,
                       lineType=cv2.LINE_AA)
            # core: blue (low) → red (high) via viridis-ish palette
            r = int(40 + 215 * v)
            g = int(40 + 60 * (1 - v))
            b = int(220 * (1 - v))
            cv2.circle(image_bgr, p, radius, (b, g, r), -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(image_bgr, p, radius, (255, 255, 255), 2,
                       lineType=cv2.LINE_AA)
        else:
            col = confidence_color(scores[coco_idx])
            cv2.circle(image_bgr, p, 8, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(image_bgr, p, 8, (255, 255, 255), 1,
                       lineType=cv2.LINE_AA)
    return image_bgr


def panel_image(sample: pd.Series, batch_dir: Path) -> tuple[np.ndarray, dict | None]:
    stem = sample["file_stem"]
    npz = np.load(batch_dir / "details" / f"{stem}.npz")
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8), None
    kpts2d = npz["kpts2d_133"]
    scores = npz["scores_133"]
    shap = joint_shapley_for_reba(sample, batch_dir)
    img = mosaic_face(img, kpts2d, scores)
    img = draw_pose_with_load(img, kpts2d, scores, shap)
    kp25_px, cf25_px = kp25_px_from_133(kpts2d, scores)
    spine = silhouette_spine_cached(
        img, kp25_px, cf25_px,
        batch_dir / "analysis" / "_spine_cache", stem,
    )
    spine = attach_node_load(spine, npz["kpts_m25"], npz["conf_m25"])
    img = draw_spine_curve_bgr(img, spine)
    img = crop_around_worker(img, kpts2d, scores)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), shap


def render_panel(ax, sample: pd.Series, img_rgb: np.ndarray,
                 domain: str, tier: str, batch_dir: Path):
    ax.imshow(img_rgb)
    ax.axis("off")

    action = int(sample.get("reba_action", 0))
    border = ACTION_COLOR_HEX.get(action, "#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(4)
        spine.set_visible(True)
    # box frame: re-enable spines after axis("off")
    ax.patch.set_edgecolor(border)
    ax.patch.set_linewidth(4)

    proc_kor = sample.get("process_label", "?")
    proc_en = _romanise_process(proc_kor)
    pred = sample.get("pred_process_id", float("nan"))
    pred_en = (_romanise_process(int(pred)) if not pd.isna(pred) else "-")
    owas = int(sample["owas_action"])
    rula = int(sample["rula_grand"])
    reba = int(sample["reba_grand"])
    biomech = sample.get("biomech_L5S1_Nm", float("nan"))
    title = (
        f"{domain.title()} ({tier.upper()})  ·  {sample['location']}\n"
        f"GT: {proc_en}  ·  Pred: {pred_en}\n"
        f"OWAS-AC={owas}   RULA={rula}   REBA={reba}"
        + (f"  ·  L5/S1={biomech:.0f} Nm" if not pd.isna(biomech) else "")
    )
    ax.set_title(title, fontsize=9, pad=4, color=border, weight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    fig_dir = batch_dir / "analysis" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # Read the most-enriched summary
    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        path = batch_dir / cand
        if path.exists():
            sum_path = path
            break
    df = pd.read_csv(sum_path)
    df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")
    log.info("Loaded %d rows from %s", len(df), sum_path)

    samples = select_load_samples(df)
    log.info("Selected %d samples", len(samples))

    domains = ["manufacturing", "eldercare", "logistics"]
    tiers = ["low", "medium", "high"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    by_key = {(s["domain"], s["tier"]): s for s in samples}
    for i, dom in enumerate(domains):
        for j, tier in enumerate(tiers):
            ax = axes[i, j]
            samp = by_key.get((dom, tier))
            if samp is None:
                ax.axis("off")
                ax.set_title(f"{dom.title()} ({tier}): no sample",
                             color="#888")
                continue
            img, shap = panel_image(samp["sample"], batch_dir)
            render_panel(ax, samp["sample"], img, dom, tier, batch_dir)

    # Title and colour-legend
    fig.suptitle(
        "Worker pose & ergonomic load across work environments: "
        "91 RTMW3D keypoints (body 17 + feet 6 + face 68; "
        "hand 91–132 excluded) + per-joint Shapley load heatmap",
        fontsize=11,
    )
    # bottom legend strip explaining colour code
    legend_ax = fig.add_axes((0.05, 0.01, 0.9, 0.03))
    legend_ax.axis("off")
    n_bins = 5
    for k in range(n_bins):
        color = plt.cm.coolwarm(k / max(n_bins - 1, 1))
        legend_ax.add_patch(
            plt.Rectangle((0.05 + k * 0.04, 0), 0.04, 1.0,
                          facecolor=color, edgecolor="none")
        )
    legend_ax.text(0.04, 0.5, "low load", ha="right", va="center",
                   fontsize=8)
    legend_ax.text(0.05 + n_bins * 0.04 + 0.01, 0.5, "high load",
                   ha="left", va="center", fontsize=8)
    # action-level legend
    x0 = 0.55
    for action in range(5):
        color = ACTION_COLOR_HEX[action]
        legend_ax.add_patch(
            plt.Rectangle((x0 + action * 0.05, 0), 0.05, 1.0,
                          facecolor=color, edgecolor="none")
        )
        legend_ax.text(x0 + action * 0.05 + 0.025, -0.4,
                       f"AL {action}\n{ACTION_LABEL[action]}",
                       ha="center", va="top", fontsize=7)

    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = fig_dir / "pose_load_overview.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Wrote %s + .pdf", out)


if __name__ == "__main__":
    main()
