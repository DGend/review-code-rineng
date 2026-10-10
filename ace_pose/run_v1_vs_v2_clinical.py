"""Validate that the BIOERA v2-CD+A backbone improves the clinical-
focus prediction over the v1 RTMW3D-only baseline on the same set of
high-risk worker poses used by Figure 17.

For each of the 16 worker poses Figure 17 selects (4 joints × 4 panels),
we:

  1. Re-load the original image.
  2. Run BIOERA v1 (single RTMW3D backbone) via the cached npz.
  3. Run BIOERA v2-CD+A (Body + BodyWithFeet + Wholebody3d ensemble)
     live on the image.
  4. For each variant, compute the per-joint external static moment
     (de Leva 1996 segment parameters).
  5. Render a side-by-side panel: left = v1 overlay + moment; right =
     v2 overlay + moment.
  6. Tabulate the per-(worker, joint) moment delta and the number of
     additionally-visible joints recovered by v2.

Output:
  * `v1_vs_v2_clinical.csv`           — full numbers
  * `v1_vs_v2_clinical_comparison.png` — 4 (joints) × 4 (panels per
                                        joint) × 2 (v1 | v2) grid
  * Console table summarising mean / median moment delta + extra
    visible joints per joint.
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

from ace_pose.biomechanics import (
    Major25Pose, compute_static_moments, _estimate_stature,
)
from ace_pose.bioera_v2 import (
    infer as v2_infer, infer_from_v1_baseline,
)
from ace_pose.run_pose_visualization import (
    mosaic_face, draw_skeleton as draw_133_skeleton, confidence_color,
)
from ace_pose.run_full_model_comparison import draw_skeleton as draw_25_skeleton
from ace_pose.run_plots import _romanise_process
from ace_pose.spine_overlay import (
    kp25_px_from_133, silhouette_spine_cached, draw_spine_curve_bgr,
    attach_node_load,
)


log = logging.getLogger("ace_pose.v1_vs_v2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


# ---------------------------------------------------------------------
# Same selection logic as the updated Figure 17
# ---------------------------------------------------------------------

JOINT_BOUNDS = {
    "L5S1_Nm":         (60.0, 300.0),
    "hip_max_Nm":      (30.0, 150.0),
    "knee_max_Nm":     (10.0, 250.0),
    "shoulder_max_Nm": (16.0,  80.0),
}
JOINT_DISPLAY = [
    ("L5/S1",   "L5S1_Nm"),
    ("Hip",     "hip_max_Nm"),
    ("Knee",    "knee_max_Nm"),
    ("Shoulder","shoulder_max_Nm"),
]


def select_panels(df_focus: pd.DataFrame, summary: pd.DataFrame,
                  n_top: int = 4) -> list[dict]:
    df = df_focus.merge(
        summary[["file_stem", "m25_conf_mean", "kpt_conf_mean"]],
        on="file_stem", how="left",
    )
    out = []
    for jname, col in JOINT_DISPLAY:
        lo, hi = JOINT_BOUNDS[col]
        cand = df[(df[col] >= lo) & (df[col] <= hi)
                   & (df["m25_conf_mean"] >= 0.60)
                   & (df["kpt_conf_mean"] >= 0.55)]
        if cand.empty:
            continue
        cand = cand.sort_values(col, ascending=False)
        p2_cut = int(len(cand) * 0.02)
        p10_cut = int(len(cand) * 0.10)
        band = cand.iloc[p2_cut: max(p2_cut + n_top, p10_cut)]
        if len(band) < n_top:
            band = cand.head(n_top)
        idx = np.linspace(0, len(band) - 1, n_top).astype(int)
        for i, ri in enumerate(idx):
            row = band.iloc[ri]
            out.append({
                "joint_name": jname, "joint_col": col,
                "v1_moment": float(row[col]),
                "file_stem": row["file_stem"],
                "domain": row["domain"],
                "process_label": row.get("process_label", "?"),
            })
    return out


# ---------------------------------------------------------------------
# Per-pose biomech for v2 ensemble output
# ---------------------------------------------------------------------

def biomech_from_kpts25(kp25_xy: np.ndarray, scores25: np.ndarray,
                       body_mass_kg: float = 70.0,
                       target_stature_m: float = 1.72) -> dict:
    """Compute external static moments at L5/S1, hip_max, knee_max,
    shoulder_max from a 2-D (25, 2) keypoint array.  We synthesize a
    3-D (25, 3) by lifting z = 0 (the static gravity moment in the
    sagittal plane depends only on horizontal moment arms — for a
    primarily-frontal-view dataset this is a reasonable
    approximation; the v1 biomech module uses the same assumption)."""
    kp25_3d = np.zeros((25, 3))
    kp25_3d[:, :2] = kp25_xy
    pose = Major25Pose(
        keypoints=kp25_3d, confidence=scores25,
        source=np.array([0]*23 + [1, 1]),
    )
    stature_pose = _estimate_stature(pose.keypoints)
    ls = target_stature_m / max(stature_pose, 1e-6) if stature_pose > 0 else 1.0
    res = compute_static_moments(pose, body_mass_kg=body_mass_kg,
                                  length_scale_m=ls)
    t = res.torque_dict()
    return {
        "L5S1_Nm":         t["L5S1"],
        "hip_max_Nm":      max(t["hip_l"], t["hip_r"]),
        "knee_max_Nm":     max(t["knee_l"], t["knee_r"]),
        "shoulder_max_Nm": max(t["shoulder_l"], t["shoulder_r"]),
    }


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------

EDGES_25 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
    (5, 23), (6, 23), (23, 24), (11, 24), (12, 24),
]


def _shared_crop_bounds(per_variant_kpts: list[tuple[np.ndarray, np.ndarray]],
                         img_shape: tuple[int, int, int]
                         ) -> tuple[int, int, int, int]:
    """Compute a single crop window that covers all keypoints from all
    variants. Each entry of `per_variant_kpts` is (kp_Nx2, conf_N)."""
    h, w = img_shape[:2]
    all_pts = []
    for kp, cf in per_variant_kpts:
        if kp is None or kp.shape[0] == 0:
            continue
        valid = cf > 0.15
        if valid.sum() >= 4:
            all_pts.append(kp[valid][:, :2])
    if not all_pts:
        return 0, 0, w, h
    all_pts = np.vstack(all_pts)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    pad_x = (x_max - x_min) * 0.12; pad_y = (y_max - y_min) * 0.12
    x0 = int(max(0, x_min - pad_x)); x1 = int(min(w, x_max + pad_x))
    y0 = int(max(0, y_min - pad_y)); y1 = int(min(h, y_max + pad_y))
    return x0, y0, x1, y1


def render_v1_panel(stem: str, batch_dir: Path,
                     crop: tuple[int, int, int, int] | None = None
                     ) -> tuple[np.ndarray, int, float, np.ndarray,
                                  np.ndarray, dict]:
    """Render v1 RTMW3D single-backbone detection using the major-25
    layout. Returns (rgb, vis25, mean_conf25, kp25_xy, cf25,
    v1_moments). v1 moments are recomputed using the SAME 2D+z=0
    convention v2 uses, so the v1 / v2 comparison isolates the
    backbone-fusion effect alone (the cached 3D moments in
    summary_with_xw.csv use a different convention)."""
    det_dir = batch_dir / "details"
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    npz = det_dir / f"{stem}.npz"
    if img is None or not npz.exists():
        return (np.zeros((100, 100, 3), dtype=np.uint8), 0, 0.0,
                np.zeros((25, 2)), np.zeros(25), {})
    data = np.load(npz)
    k2 = data["kpts2d_133"]; sc = data["scores_133"]
    kp25_xy = data["kpts_m25"][:, :2].astype(float)
    cf25 = data["conf_m25"].astype(float)
    img = mosaic_face(img.copy(), k2, sc)
    img = draw_25_skeleton(img.copy(), kp25_xy, cf25, EDGES_25,
                            body_radius=6, edge_thick=4)
    kp25_px, cf25_px = kp25_px_from_133(k2, sc)
    spine = silhouette_spine_cached(
        img, kp25_px, cf25_px,
        batch_dir / "analysis" / "_spine_cache", stem,
    )
    spine = attach_node_load(spine, data["kpts_m25"], cf25)
    img = draw_spine_curve_bgr(img, spine)
    if crop is None:
        x0, y0, x1, y1 = _shared_crop_bounds([(kp25_xy, cf25)], img.shape)
    else:
        x0, y0, x1, y1 = crop
    img = img[y0:y1, x0:x1]
    vis_25 = int((cf25 >= 0.5).sum())
    mean_25 = float(cf25.mean())
    # Recompute v1 moments using the same 2D+z=0 convention as v2
    v1_moments = biomech_from_kpts25(kp25_xy, cf25)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             vis_25, mean_25, kp25_xy, cf25, v1_moments)


def render_v2_panel(stem: str, batch_dir: Path,
                    variant: str = "v2-CD+A",
                    crop: tuple[int, int, int, int] | None = None,
                    target_centroid: np.ndarray | None = None,
                    v1_baseline: tuple[np.ndarray, np.ndarray] | None = None,
                    ) -> tuple[np.ndarray, dict, int, float, dict,
                                 np.ndarray, np.ndarray]:
    """Run the named v2 variant live on the image, return rendered
    crop + per-joint moments + visibility metrics + (kpts, scores)
    for shared-crop computation."""
    det_dir = batch_dir / "details"
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    if img is None:
        return (np.zeros((100, 100, 3), dtype=np.uint8), {}, 0, 0.0, {},
                np.zeros((25, 2)), np.zeros(25))
    # Face mosaic from cached v1 RTMW3D 133-pt face landmarks (we keep
    # the privacy mask identical across variants for fair comparison).
    npz = det_dir / f"{stem}.npz"
    if npz.exists():
        data = np.load(npz)
        k2 = data["kpts2d_133"]; sc = data["scores_133"]
        img = mosaic_face(img.copy(), k2, sc)
    # For v2-guarded, use the v1 cached positions as an EXPLICIT spine
    # (no live RTMW3D rerun) to guarantee person identity matches v1.
    if variant == "v2-guarded" and v1_baseline is not None:
        v1_kp, v1_cf = v1_baseline
        res = infer_from_v1_baseline(img, v1_kp, v1_cf,
                                      guard_thresh=0.50,
                                      guard_bbox_px=40.0)
    else:
        res = v2_infer(img, variant=variant,
                        target_centroid=target_centroid)
    if res is None:
        return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), {}, 0, 0.0, {},
                np.zeros((25, 2)), np.zeros(25))
    kpts = res["kpts"]; scs = res["scores"]
    img = draw_25_skeleton(img.copy(), kpts, scs, res["edges"],
                            body_radius=6, edge_thick=4)
    if npz.exists():
        kp25_px, cf25_px = kp25_px_from_133(k2, sc)
        spine = silhouette_spine_cached(
            img, kp25_px, cf25_px,
            batch_dir / "analysis" / "_spine_cache", stem,
        )
        spine = attach_node_load(spine, data["kpts_m25"],
                                 data["conf_m25"])
        img = draw_spine_curve_bgr(img, spine)
    moments = biomech_from_kpts25(kpts[:, :2], scs)
    vis_25 = int((scs >= 0.5).sum())
    mean_25 = float(scs.mean())
    if crop is None:
        x0, y0, x1, y1 = _shared_crop_bounds([(kpts, scs)], img.shape)
    else:
        x0, y0, x1, y1 = crop
    img = img[y0:y1, x0:x1]
    sigma_mean = (float(np.nanmean(res["simcc_sigma"][:25]))
                  if res["simcc_sigma"] is not None else float("nan"))
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), moments,
            vis_25, mean_25, {"sigma_cross": sigma_mean},
            kpts, scs)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--n_per_joint", type=int, default=4)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    out_dir = batch_dir / "analysis"
    fig_dir = out_dir / "figures"

    summary = pd.read_csv(batch_dir / "summary_with_xw.csv")
    focus = pd.read_csv(out_dir / "joint_focus.csv")
    panels = select_panels(focus, summary, n_top=args.n_per_joint)
    log.info("Selected %d (joint, worker) panels", len(panels))

    # Five fusion modes compared side-by-side.
    VARIANTS = [
        ("v1",          "v1 (RTMW3D-only)"),
        ("v2-CD-soft",  "v2-soft (legacy)"),
        ("v2-CD-max",   "v2-max"),
        ("v2-sigma",    "v2-σ (v1 pos)"),
        ("v2-guarded",  "v2-guarded (rec.)"),
    ]

    rows = []
    rendered: list[dict] = []
    for p in panels:
        stem = p["file_stem"]
        # First pass — gather all variant keypoints, compute shared crop
        _, _, _, v1_kp, v1_cf, v1_moms = render_v1_panel(stem, batch_dir,
                                                          crop=None)
        # v1 primary worker centroid from cached body-17 joints — used
        # to lock the v2 ensemble onto the same person
        v1_valid_body = v1_cf[:17] > 0.15
        if v1_valid_body.sum() >= 3:
            v1_centroid = v1_kp[:17][v1_valid_body, :2].mean(axis=0)
        else:
            v1_centroid = None
        img_probe = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        img_shape = img_probe.shape if img_probe is not None else (1, 1, 3)
        variant_kpts = [(v1_kp, v1_cf)]
        v2_outs = {}
        v1_baseline = (v1_kp[:, :2], v1_cf)
        for var_id, _label in VARIANTS[1:]:
            (rgb_, moms_, vis_, meanc_, extra_,
             kp_, sc_) = render_v2_panel(stem, batch_dir, variant=var_id,
                                          crop=None,
                                          target_centroid=v1_centroid,
                                          v1_baseline=v1_baseline)
            v2_outs[var_id] = (moms_, vis_, meanc_, extra_, kp_, sc_)
            variant_kpts.append((kp_, sc_))
        shared = _shared_crop_bounds(variant_kpts, img_shape)

        # Second pass — re-render with the shared crop window
        v1_rgb, v1_vis, v1_meanc, _, _, _ = render_v1_panel(
            stem, batch_dir, crop=shared,
        )
        # Use the RECOMPUTED v1 moments (2D+z=0 convention) — same as
        # v2 — for fair fusion comparison. Update the panel record's
        # baseline v1 moment to reflect this.
        p["v1_moment"] = v1_moms.get(p["joint_col"], p["v1_moment"])
        per_variant = {"v1": (v1_rgb, v1_moms,
                              v1_vis, v1_meanc, {})}
        for var_id, _label in VARIANTS[1:]:
            (rgb, moms, vis, meanc, extra, _kp,
             _sc) = render_v2_panel(stem, batch_dir, variant=var_id,
                                      crop=shared,
                                      target_centroid=v1_centroid,
                                      v1_baseline=v1_baseline)
            per_variant[var_id] = (rgb, moms, vis, meanc, extra)

        v1_mom = p["v1_moment"]
        row = {
            "joint_name":   p["joint_name"], "joint_col": p["joint_col"],
            "file_stem":    stem, "domain": p["domain"],
            "process_label": _romanise_process(p["process_label"]),
            "v1_moment_Nm": v1_mom,
            "v1_vis_25": v1_vis,    "v1_mean_conf": v1_meanc,
        }
        for var_id, _label in VARIANTS[1:]:
            rgb, moms, vis, meanc, extra = per_variant[var_id]
            mom = moms.get(p["joint_col"], float("nan"))
            row[f"{var_id}_moment_Nm"] = mom
            row[f"{var_id}_delta_Nm"]  = mom - v1_mom
            row[f"{var_id}_vis_25"]    = vis
            row[f"{var_id}_mean_conf"] = meanc
            row[f"{var_id}_sigma_cross_px"] = extra.get("sigma_cross",
                                                         float("nan"))
        rows.append(row)
        rendered.append({"panel": p, "per_variant": per_variant})

    out_csv = out_dir / "v1_vs_v2_clinical.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    df = pd.DataFrame(rows)
    agg = {"n": ("file_stem", "count"),
           "v1_mean": ("v1_moment_Nm", "mean"),
           "v1_vis_mean": ("v1_vis_25", "mean"),
           "v1_conf_mean": ("v1_mean_conf", "mean")}
    for var_id, _ in VARIANTS[1:]:
        agg[f"{var_id}_mean"]      = (f"{var_id}_moment_Nm", "mean")
        agg[f"{var_id}_delta"]     = (f"{var_id}_delta_Nm",  "mean")
        agg[f"{var_id}_vis_mean"]  = (f"{var_id}_vis_25",    "mean")
        agg[f"{var_id}_conf_mean"] = (f"{var_id}_mean_conf", "mean")
    summary_tbl = df.groupby("joint_name").agg(**agg).round(2)
    log.info("\n4-way per-joint summary:\n%s", summary_tbl.to_string())
    summary_tbl.to_csv(out_dir / "v1_vs_v2_clinical_summary.csv")

    # Render figure: 4 rows (joints) × 4 cols (fusion modes), one
    # representative worker per joint (the highest-moment plausible
    # case selected by the p2-p10 band).
    ncols = len(VARIANTS)
    fig, axes = plt.subplots(len(JOINT_DISPLAY), ncols,
                              figsize=(3.6 * ncols, 4.4 * len(JOINT_DISPLAY)))
    for ri, (jname, col) in enumerate(JOINT_DISPLAY):
        joint_workers = [r for r in rendered
                          if r["panel"]["joint_name"] == jname]
        if not joint_workers:
            for ci in range(ncols):
                axes[ri, ci].axis("off")
            continue
        # Pick the worker where v2-CD-soft moves the moment most vs v1
        # — this surfaces the fusion artefact rather than averaging it
        # out. Falls back to the highest-moment worker if soft Δ ties.
        def _soft_delta(item):
            v1m = item["panel"]["v1_moment"]
            soft_moms = item["per_variant"]["v2-CD-soft"][1]
            sm = soft_moms.get(col, float("nan"))
            return abs(sm - v1m) if np.isfinite(sm) else 0.0
        joint_workers = sorted(joint_workers, key=_soft_delta, reverse=True)
        item = joint_workers[0]                # max |soft Δ|
        p = item["panel"]
        proc = _romanise_process(p["process_label"])
        for ci, (var_id, label) in enumerate(VARIANTS):
            ax = axes[ri, ci]
            rgb, moms, vis, meanc, extra = item["per_variant"][var_id]
            ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
            mom = (p["v1_moment"] if var_id == "v1"
                    else moms.get(col, float("nan")))
            delta = (mom - p["v1_moment"]) if var_id != "v1" else 0.0
            title_extra = f"  (Δ{delta:+.0f})" if var_id != "v1" else ""
            colour = "#dc2626" if (var_id == "v2-CD-soft"
                                    and abs(delta) > 25) else "black"
            ax.set_title(f"{label}\n{mom:.0f} Nm{title_extra}",
                          fontsize=10.5, weight="bold", pad=4,
                          color=colour)
            ax.set_xlabel(f"vis = {vis}/25   mean conf = {meanc:.2f}",
                           fontsize=8.5, labelpad=2)
        axes[ri, 0].set_ylabel(
            f"{jname}\n{p['domain']}\n{proc}",
            fontsize=11, weight="bold", rotation=0,
            ha="right", va="center", labelpad=24,
        )

    fig.suptitle(
        "BIOERA v1 → v2 fusion-mode comparison on the single highest-moment "
        "worker per surgical-risk joint.\n"
        "Columns: v1 (single RTMW3D backbone) │ "
        "v2-CD-soft (confidence-weighted soft average — legacy, blurs the "
        "whole-body prior) │ "
        "v2-CD-max (winner-takes-all — recommended) │ "
        "v2-σ (v1 positions + cross-backbone uncertainty signal only).\n"
        "All four panels of a row use the IDENTICAL major-25 visualisation "
        "and the SAME shared crop window — differences reflect backbone "
        "fusion choice alone.  Red v2-CD-soft title = the soft-average "
        "moved the moment by more than 25 Nm vs v1, i.e. fusion artefact.",
        fontsize=11.5, weight="bold", y=0.999,
    )
    fig.tight_layout(rect=(0.06, 0, 1, 0.92))
    out_png = fig_dir / "v1_vs_v2_clinical_comparison.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_png)


if __name__ == "__main__":
    main()
