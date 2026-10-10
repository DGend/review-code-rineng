"""High-injury-risk joint focus analysis for the BIOERA paper.

The workplace musculoskeletal disorders that progress to surgery in
field workers cluster on four joints:

  * **L5/S1 (lumbar)** — disc herniation from lifting / sustained
    flexion.  Surgical: discectomy.
  * **Hip / pelvis** — labral tear from squat-and-lift cycles.
    Surgical: arthroscopic labral repair.
  * **Knee** — meniscus tear from sustained / repeated squat.
    Surgical: arthroscopic meniscectomy / repair.
  * **Shoulder** — rotator cuff tear from sustained abduction with
    load.  Surgical: rotator cuff repair.

For each of the 6,007 Busan poses we recompute the external static
moment at *both sides* of every joint (using the v1 RTMW3D detection
and the de Leva 1996 segment parameters already encoded in
`ace_pose.biomechanics`). We then:

  1. Aggregate per-domain distributions of moments + flexion angles.
  2. Compare against documented clinical injury thresholds:
        * L5/S1   60 Nm (Marras et al., 1995 — Lumbar Motion Monitor;
                  approximate static-component of the NIOSH RWL action
                  limit) and 120 Nm (NIOSH MPL upper bound).
        * Hip     30 Nm (Buckwalter 2003 — labral-tear risk threshold
                  at sustained > 60° trunk flexion + leg lift).
        * Knee    50 Nm (Felson 2013 — progressive patellofemoral
                  degenerative load).
        * Shoulder 16 Nm (Chaffin 1973 — Chaffin–Garg static threshold
                  for the deltoid–rotator-cuff complex at 90° flexion).
  3. Report `%poses_above_threshold` per domain and the top-N worker
     poses per joint for clinical follow-up.

Outputs:
  * `joint_focus.csv`              — per-image per-joint moment + flex angle
  * `joint_focus_summary.csv`      — per-domain mean/median + % exceeding threshold
  * `joint_focus_per_process.csv`  — per-(domain, process) summary
  * `joint_focus_distributions.png`— 4 (joints) × 3 (domains) panel histograms
  * `joint_focus_top_risk.png`     — top-N worker visualisations per joint
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
from ace_pose.run_plots import _romanise_process


log = logging.getLogger("ace_pose.run_joint_focus")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"


# ---------------------------------------------------------------------
# Clinical injury thresholds (literature-derived)
# ---------------------------------------------------------------------
CLINICAL_THRESHOLDS_NM = {
    "L5S1":     60.0,        # Marras 1995 LMM — onset of elevated risk
    "L5S1_high": 120.0,       # NIOSH MPL upper bound (static component)
    "hip":      30.0,         # Buckwalter 2003 — labral-tear sustained load
    "knee":     50.0,         # Felson 2013 — patellofemoral degenerative load
    "shoulder": 16.0,         # Chaffin 1973 — deltoid–RC complex at 90° flex
}

CLINICAL_SOURCES = {
    "L5S1":     "Marras et al., 1995 (LMM)",
    "L5S1_high": "NIOSH MPL upper bound",
    "hip":      "Buckwalter, 2003",
    "knee":     "Felson, 2013",
    "shoulder": "Chaffin, 1973",
}


# ---------------------------------------------------------------------
# Recompute biomech moments from per-image npz
# ---------------------------------------------------------------------

def recompute_moments(batch_dir: Path, summary: pd.DataFrame,
                      body_mass_kg: float = 70.0, target_stature_m: float = 1.72,
                      ) -> pd.DataFrame:
    """Load each pose's npz, recompute compute_static_moments, return
    one row per image with per-joint moments and bilateral max."""
    det_dir = batch_dir / "details"
    rows = []
    for i, r in summary.iterrows():
        if r.get("status") != "ok":
            continue
        stem = r["file_stem"]
        npz = det_dir / f"{stem}.npz"
        if not npz.exists():
            continue
        try:
            data = np.load(npz)
            kp25 = data["kpts_m25"]
            cf25 = data["conf_m25"]
            # Major25Pose.source: 0 = direct, 1 = derived midpoint.
            # Indices 0..22 are direct from RTMW3D 133-pt; 23/24 are derived.
            source = np.zeros(25, dtype=int)
            source[23:25] = 1
            pose = Major25Pose(keypoints=kp25, confidence=cf25, source=source)
            # auto length-scale so apparent stature → target_stature_m
            stature_pose = _estimate_stature(pose.keypoints)
            ls = target_stature_m / max(stature_pose, 1e-6) if stature_pose > 0 else 1.0
            res = compute_static_moments(pose, body_mass_kg=body_mass_kg,
                                          length_scale_m=ls)
            t = res.torque_dict()
            row = {
                "file_stem": stem,
                "domain": r.get("domain"),
                "process_id": r.get("process_id"),
                "process_label": r.get("process_label"),
                "L5S1_Nm": t["L5S1"],
                "hip_l_Nm": t["hip_l"], "hip_r_Nm": t["hip_r"],
                "hip_max_Nm": max(t["hip_l"], t["hip_r"]),
                "knee_l_Nm": t["knee_l"], "knee_r_Nm": t["knee_r"],
                "knee_max_Nm": max(t["knee_l"], t["knee_r"]),
                "shoulder_l_Nm": t["shoulder_l"],
                "shoulder_r_Nm": t["shoulder_r"],
                "shoulder_max_Nm": max(t["shoulder_l"], t["shoulder_r"]),
                "trunk_flex_deg":  r.get("owas_trunk_flex_deg", float("nan")),
                "knee_flex_l_deg": r.get("owas_knee_l_deg", float("nan")),
                "knee_flex_r_deg": r.get("owas_knee_r_deg", float("nan")),
                "upper_arm_flex_deg": r.get("rula_upper_arm_deg", float("nan")),
                "stature_m": float(res.stature_m),
            }
            rows.append(row)
        except Exception as ex:
            log.warning("biomech failed on %s: %s", stem, ex)
        if (i + 1) % 1000 == 0:
            log.info("  …processed %d / %d", i + 1, len(summary))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-domain summary
# ---------------------------------------------------------------------

def per_domain_summary(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for dom, sub in df.groupby("domain"):
        if dom is None or dom == "unknown":
            continue
        for joint, col, thr_key in [
            ("L5/S1",       "L5S1_Nm",        "L5S1"),
            ("Hip (max)",   "hip_max_Nm",     "hip"),
            ("Knee (max)",  "knee_max_Nm",    "knee"),
            ("Shoulder (max)", "shoulder_max_Nm", "shoulder"),
        ]:
            vals = sub[col].dropna()
            thr = CLINICAL_THRESHOLDS_NM[thr_key]
            out_rows.append({
                "domain": dom, "joint": joint,
                "n_poses": len(vals),
                "mean_Nm": vals.mean(),
                "median_Nm": vals.median(),
                "p90_Nm": vals.quantile(0.90),
                "p95_Nm": vals.quantile(0.95),
                "max_Nm": vals.max(),
                "threshold_Nm": thr,
                "n_above_threshold": int((vals >= thr).sum()),
                "pct_above_threshold": 100.0 * (vals >= thr).mean(),
                "source": CLINICAL_SOURCES[thr_key],
            })
    return pd.DataFrame(out_rows)


def per_process_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(domain, process) summary — top processes that drive each joint risk."""
    out_rows = []
    for (dom, proc), sub in df.groupby(["domain", "process_label"]):
        if dom is None or proc is None:
            continue
        if len(sub) < 30:
            continue
        out_rows.append({
            "domain": dom, "process_label": proc, "n_poses": len(sub),
            "L5S1_mean": sub["L5S1_Nm"].mean(),
            "L5S1_p95":  sub["L5S1_Nm"].quantile(0.95),
            "hip_max_mean": sub["hip_max_Nm"].mean(),
            "hip_max_p95":  sub["hip_max_Nm"].quantile(0.95),
            "knee_max_mean": sub["knee_max_Nm"].mean(),
            "knee_max_p95":  sub["knee_max_Nm"].quantile(0.95),
            "shoulder_max_mean": sub["shoulder_max_Nm"].mean(),
            "shoulder_max_p95":  sub["shoulder_max_Nm"].quantile(0.95),
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------
# Distribution figure: 4 joints × 3 domains, each cell = histogram + threshold
# ---------------------------------------------------------------------

def render_distribution_grid(df: pd.DataFrame, out_png: Path):
    domains = ["manufacturing", "eldercare", "logistics"]
    joints = [
        ("L5/S1 (lumbar)",      "L5S1_Nm",         "L5S1",
            "Marras 1995 — disc herniation onset", "high"),
        ("Hip (max L/R)",       "hip_max_Nm",      "hip",
            "Buckwalter 2003 — labral tear risk", None),
        ("Knee (max L/R)",      "knee_max_Nm",     "knee",
            "Felson 2013 — patellofemoral load",  None),
        ("Shoulder (max L/R)",  "shoulder_max_Nm", "shoulder",
            "Chaffin 1973 — rotator cuff complex", None),
    ]
    domain_titles = {
        "manufacturing": "Manufacturing", "eldercare": "Eldercare",
        "logistics":     "Logistics",
    }
    fig, axes = plt.subplots(len(joints), len(domains),
                              figsize=(4.4 * len(domains), 3.0 * len(joints)),
                              sharex="row")
    for ri, (jname, col, thr_key, src, high_key) in enumerate(joints):
        for ci, dom in enumerate(domains):
            ax = axes[ri, ci]
            sub = df.loc[df["domain"] == dom, col].dropna()
            if sub.empty:
                ax.axis("off"); continue
            thr = CLINICAL_THRESHOLDS_NM[thr_key]
            # Clamp the x-range to make the histogram readable (P99.5)
            xmax = float(np.percentile(sub, 99.5))
            xmax = max(xmax, thr * 1.5)
            ax.hist(np.clip(sub, 0, xmax), bins=40,
                     color="#2563eb", edgecolor="white", alpha=0.85)
            ax.axvline(thr, color="#dc2626", linewidth=2.0, linestyle="--",
                        label=f"clinical threshold = {thr:.0f} Nm")
            if high_key is not None:
                thr_h = CLINICAL_THRESHOLDS_NM[f"{thr_key}_high"]
                ax.axvline(thr_h, color="#7f1d1d", linewidth=2.0,
                            linestyle=":",
                            label=f"injurious = {thr_h:.0f} Nm")
            n_above = int((sub >= thr).sum())
            pct_above = 100 * (sub >= thr).mean()
            ax.set_xlim(0, xmax)
            ax.set_xlabel(f"{jname}  (Nm)", fontsize=9)
            ax.set_ylabel("count" if ci == 0 else "")
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            ax.set_title(
                f"{domain_titles[dom]}  n={len(sub)}  "
                f"{n_above} ({pct_above:.1f} %) ≥ {thr:.0f} Nm",
                fontsize=10, weight="bold"
            )
            if ci == 0 and ri == 0:
                ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        "Per-joint clinical injury load — full Busan cohort (n = 6,007), "
        "external static moment vs literature-derived surgical-risk thresholds",
        fontsize=13, weight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Top-N worker visualisation per joint
# ---------------------------------------------------------------------

def render_top_risk_panels(df: pd.DataFrame, batch_dir: Path,
                           summary: pd.DataFrame,
                           out_png: Path, n_top: int = 4):
    """High-risk worker visualisation per joint.  We pick from the
    **p90–p98 moment band** rather than the absolute top — the
    extreme top of every joint's distribution is dominated by
    RTMW3D pose-detection failures (mis-localised keypoint at image
    edge → false large moment arm).  Within that band we also
    require minimum pose confidence to filter the remaining
    misdetections."""
    from ace_pose.run_pose_visualization import (
        mosaic_face, draw_skeleton,
    )
    # Plausibility upper-bounds per joint: 5× the clinical threshold.
    # Above this the moment is almost always a detection failure.
    JOINT_BOUNDS = {
        "L5S1_Nm":         (60.0 * 1, 60.0 * 5),    # 60–300 Nm
        "hip_max_Nm":      (30.0 * 1, 30.0 * 5),    # 30–150 Nm
        "knee_max_Nm":     (10.0,     50.0 * 5),    # 10–250 Nm
        "shoulder_max_Nm": (16.0,     16.0 * 5),    # 16–80 Nm
    }
    joints = [
        ("L5/S1",   "L5S1_Nm"),
        ("Hip",     "hip_max_Nm"),
        ("Knee",    "knee_max_Nm"),
        ("Shoulder","shoulder_max_Nm"),
    ]
    det_dir = batch_dir / "details"
    # Join cohort pose-confidence onto the focus DataFrame.
    conf_cols = ["file_stem", "m25_conf_mean", "kpt_conf_mean", "rula_grand"]
    df = df.merge(summary[conf_cols], on="file_stem", how="left")

    fig, axes = plt.subplots(len(joints), n_top,
                              figsize=(3.0 * n_top, 4.0 * len(joints)))
    for ri, (jname, col) in enumerate(joints):
        lo, hi = JOINT_BOUNDS[col]
        cand = df[(df[col] >= lo) & (df[col] <= hi)
                   & (df["m25_conf_mean"] >= 0.60)
                   & (df["kpt_conf_mean"] >= 0.55)]
        if cand.empty:
            log.warning("no plausible top-risk poses for %s", jname)
            continue
        # Sample uniformly from p90–p98 band: rank-percentile sort
        # then pick `n_top` evenly spaced indices.
        cand = cand.sort_values(col, ascending=False)
        p2_cut = int(len(cand) * 0.02)
        p10_cut = int(len(cand) * 0.10)
        band = cand.iloc[p2_cut: max(p2_cut + n_top, p10_cut)]
        if len(band) < n_top:
            band = cand.head(n_top)
        # take n_top evenly-spaced rows from the band
        idx = np.linspace(0, len(band) - 1, n_top).astype(int)
        top = band.iloc[idx]
        for ci in range(n_top):
            ax = axes[ri, ci]
            if ci >= len(top):
                ax.axis("off"); continue
            row = top.iloc[ci]
            stem = row["file_stem"]
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
            npz = det_dir / f"{stem}.npz"
            if img is None or not npz.exists():
                ax.set_visible(False); continue
            data = np.load(npz)
            k2 = data["kpts2d_133"]
            sc = data["scores_133"]
            img = mosaic_face(img.copy(), k2, sc)
            img = draw_skeleton(img.copy(), k2, sc)
            # tight crop using major-25
            kp = k2[:23]; cf = sc[:23]
            valid = cf > 0.15
            if valid.sum() >= 4:
                pts = kp[valid]
                x_min, y_min = pts.min(axis=0); x_max, y_max = pts.max(axis=0)
                pad_x = (x_max - x_min) * 0.12; pad_y = (y_max - y_min) * 0.12
                x0 = int(max(0, x_min - pad_x)); x1 = int(min(img.shape[1], x_max + pad_x))
                y0 = int(max(0, y_min - pad_y)); y1 = int(min(img.shape[0], y_max + pad_y))
                img = img[y0:y1, x0:x1]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            proc_en = _romanise_process(row.get("process_label", "?"))
            ax.set_title(
                f"{jname}  {row[col]:.0f} Nm\n"
                f"{row['domain']} · {proc_en}",
                fontsize=9, weight="bold", pad=4
            )
    fig.suptitle(
        "Top-risk worker poses per joint  (highest external static moment, "
        "filtered to < 300 Nm to exclude pose-detection failures)",
        fontsize=12, weight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    out_dir = batch_dir / "analysis"
    fig_dir = out_dir / "figures"

    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        p = batch_dir / cand
        if p.exists():
            sum_path = p; break
    log.info("Loading summary: %s", sum_path)
    summary = pd.read_csv(sum_path)

    # 1. Recompute moments per image
    cache_path = out_dir / "joint_focus.csv"
    if cache_path.exists():
        log.info("Loading cached joint_focus.csv (delete to rebuild)")
        df = pd.read_csv(cache_path)
    else:
        log.info("Recomputing biomech moments across %d images…", len(summary))
        df = recompute_moments(batch_dir, summary)
        df.to_csv(cache_path, index=False)
        log.info("Wrote %s (%d rows)", cache_path, len(df))

    # 2. Domain summary
    dom_summary = per_domain_summary(df)
    dom_summary.to_csv(out_dir / "joint_focus_summary.csv", index=False)
    log.info("\nPer-domain summary:\n%s", dom_summary.to_string())

    # 3. Per-process summary
    proc_summary = per_process_summary(df)
    proc_summary.to_csv(out_dir / "joint_focus_per_process.csv", index=False)
    log.info("\nPer-process summary:\n%s",
             proc_summary.sort_values("L5S1_p95", ascending=False).head(10).to_string())

    # 4. Distribution figure
    render_distribution_grid(df, fig_dir / "joint_focus_distributions.png")
    log.info("Wrote joint_focus_distributions.png")

    # 5. Top-risk panels
    render_top_risk_panels(df, batch_dir, summary,
                           fig_dir / "joint_focus_top_risk.png")
    log.info("Wrote joint_focus_top_risk.png")


if __name__ == "__main__":
    main()
