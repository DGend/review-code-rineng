"""Per-vertebral-level dense-spine analysis over the full Busan
cohort (6,007 poses).

For every pose we compute the 9-level dense spine (sacrum, L5/S1,
L3, L1, T12, T6, T1, C7, occiput) via anatomical interpolation
(`ace_pose.dense_spine.estimate_dense_spine`) — *no learned model*,
sub-millisecond per pose. Per-level flexion and external moment are
exported, then aggregated by domain to identify which **vertebral
level** carries the highest load in each work environment.

Output:
  * `dense_spine.csv`              — per-image × per-level metrics
  * `dense_spine_summary.csv`      — per-domain × per-level statistics
  * `dense_spine_distribution.png` — 3 domains × 9 levels heatmap of
                                     mean flexion / mean moment
  * `dense_spine_focus.png`        — L5/S1 vs L3 vs T6 flexion +
                                     moment distributions per domain
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.dense_spine import (
    estimate_dense_spine, headline_spine_metrics, SPINE_LEVELS,
)
from ace_pose.biomechanics import _estimate_stature


log = logging.getLogger("ace_pose.run_dense_spine")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def recompute_spine(batch_dir: Path, summary: pd.DataFrame,
                    body_mass_kg: float = 70.0,
                    target_stature_m: float = 1.72,
                    ) -> pd.DataFrame:
    """Compute dense spine per image. Returns wide-format DataFrame:
    one row per image, with `flex_<level>_deg`, `mom_<level>_Nm`
    columns per spinal level."""
    det_dir = batch_dir / "details"
    rows = []
    level_names = [name for name, _, _ in SPINE_LEVELS]
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
            # length scale → target stature 1.72 m
            stature_pose = _estimate_stature(kp25)
            ls = target_stature_m / max(stature_pose, 1e-6) if stature_pose > 0 else 1.0
            s = estimate_dense_spine(kp25, cf25,
                                      body_mass_kg=body_mass_kg,
                                      length_scale_m=ls)
            row = {
                "file_stem": stem,
                "domain": r.get("domain"),
                "process_id": r.get("process_id"),
                "process_label": r.get("process_label"),
            }
            for j, name in enumerate(level_names):
                row[f"flex_{name}_deg"] = float(s.flex_deg[j])
                row[f"mom_{name}_Nm"] = float(s.moments_Nm[j])
            # headlines
            row.update(headline_spine_metrics(s))
            rows.append(row)
        except Exception as ex:
            log.warning("spine failed on %s: %s", stem, ex)
        if (i + 1) % 1000 == 0:
            log.info("  …processed %d / %d", i + 1, len(summary))
    return pd.DataFrame(rows)


def per_domain_per_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form (domain, level) summary."""
    level_names = [name for name, _, _ in SPINE_LEVELS]
    rows = []
    for dom, sub in df.groupby("domain"):
        if dom is None or dom == "unknown":
            continue
        for name in level_names:
            f = sub[f"flex_{name}_deg"].dropna()
            m = sub[f"mom_{name}_Nm"].dropna()
            rows.append({
                "domain": dom, "level": name,
                "n_poses": len(f),
                "flex_mean_deg": f.mean(), "flex_p90_deg": f.quantile(0.9),
                "flex_p95_deg": f.quantile(0.95), "flex_max_deg": f.max(),
                "mom_mean_Nm": m.mean(), "mom_p90_Nm": m.quantile(0.9),
                "mom_p95_Nm": m.quantile(0.95), "mom_max_Nm": m.max(),
                # Clinical flex thresholds: > 30° = onset, > 60° = high risk
                "pct_above_30_deg": 100 * (f >= 30).mean(),
                "pct_above_60_deg": 100 * (f >= 60).mean(),
            })
    return pd.DataFrame(rows)


def render_heatmap(df_summary: pd.DataFrame, out_png: Path):
    """3 domains × 9 levels heatmap of mean flexion + mean moment."""
    level_names = [name for name, _, _ in SPINE_LEVELS]
    domains = ["manufacturing", "eldercare", "logistics"]
    flex_mat = np.zeros((len(domains), len(level_names)))
    mom_mat = np.zeros((len(domains), len(level_names)))
    for di, dom in enumerate(domains):
        for li, lev in enumerate(level_names):
            sub = df_summary[(df_summary["domain"] == dom)
                              & (df_summary["level"] == lev)]
            if len(sub) == 0:
                continue
            flex_mat[di, li] = float(sub["flex_mean_deg"].iloc[0])
            mom_mat[di, li]  = float(sub["mom_mean_Nm"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, mat, label, cmap in [
        (axes[0], flex_mat, "Mean trunk flexion at level (degrees)", "Reds"),
        (axes[1], mom_mat,  "Mean external moment at level (Nm)",   "Blues"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels(level_names, rotation=0, fontsize=10)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels([d.title() for d in domains], fontsize=10)
        ax.set_title(label, fontsize=11, weight="bold")
        for di in range(mat.shape[0]):
            for li in range(mat.shape[1]):
                val = mat[di, li]
                colour = "white" if val > mat.max() * 0.55 else "black"
                ax.text(li, di, f"{val:.1f}", ha="center", va="center",
                         color=colour, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Dense Spine Estimation — per-vertebral-level mean trunk flexion (top) "
        "and external static moment (bottom) by work domain (n = 6,007)",
        fontsize=13, weight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_focus_levels(df: pd.DataFrame, out_png: Path):
    """Compare L5/S1 vs L3 vs T6 — the three clinically critical
    levels — across domains."""
    levels = ["L5/S1", "L3", "T6"]
    cols_flex = [f"flex_{lev}_deg" for lev in levels]
    cols_mom = [f"mom_{lev}_Nm"  for lev in levels]
    domains = ["manufacturing", "eldercare", "logistics"]
    colours = {"manufacturing": "#2563eb", "eldercare": "#dc2626",
               "logistics": "#16a34a"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="row")
    for ci, lev in enumerate(levels):
        # Flexion row
        axf = axes[0, ci]
        for dom in domains:
            vals = df.loc[df["domain"] == dom, f"flex_{lev}_deg"].dropna()
            if vals.empty:
                continue
            axf.hist(np.clip(vals, 0, 120), bins=40, alpha=0.55,
                      color=colours[dom], label=dom.title(),
                      edgecolor="white")
        axf.axvline(30, color="#facc15", linewidth=1.5, linestyle=":",
                     label="onset 30°")
        axf.axvline(60, color="#dc2626", linewidth=1.5, linestyle="--",
                     label="high-risk 60°")
        axf.set_xlim(0, 120)
        axf.set_title(f"{lev} sagittal flexion (°)", fontsize=11, weight="bold")
        axf.set_xlabel(f"flexion at {lev} (°)")
        axf.set_ylabel("count" if ci == 0 else "")
        axf.grid(axis="y", linestyle=":", alpha=0.3)
        if ci == 0:
            axf.legend(loc="upper right", fontsize=8)

        # Moment row
        axm = axes[1, ci]
        for dom in domains:
            vals = df.loc[df["domain"] == dom, f"mom_{lev}_Nm"].dropna()
            if vals.empty:
                continue
            xmax = float(np.percentile(vals.values, 99.5))
            xmax = max(xmax, 150.0)
            axm.hist(np.clip(vals, 0, xmax), bins=40, alpha=0.55,
                      color=colours[dom], edgecolor="white")
        axm.axvline(60, color="#dc2626", linewidth=1.5, linestyle="--",
                     label="clinical 60 Nm")
        axm.set_xlim(0, 250)
        axm.set_title(f"{lev} external moment (Nm)", fontsize=11, weight="bold")
        axm.set_xlabel(f"moment at {lev} (Nm)")
        axm.set_ylabel("count" if ci == 0 else "")
        axm.grid(axis="y", linestyle=":", alpha=0.3)
        if ci == 0:
            axm.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Clinically critical spinal levels — L5/S1 (disc herniation), "
        "L3 (mid-lumbar fatigue), T6 (thoracic kyphosis): flexion (top) and "
        "external moment (bottom) distributions per work domain",
        fontsize=13, weight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
    log.info("Loading %s", sum_path)
    summary = pd.read_csv(sum_path)

    cache_path = out_dir / "dense_spine.csv"
    if cache_path.exists():
        log.info("Loading cached dense_spine.csv (delete to rebuild)")
        df = pd.read_csv(cache_path)
    else:
        log.info("Computing dense spine across %d images…", len(summary))
        df = recompute_spine(batch_dir, summary)
        df.to_csv(cache_path, index=False)
        log.info("Wrote %s (%d rows)", cache_path, len(df))

    log.info("Per-domain × per-level summary…")
    dom_summary = per_domain_per_level_summary(df)
    dom_summary.to_csv(out_dir / "dense_spine_summary.csv", index=False)
    log.info("\n%s", dom_summary.to_string())

    render_heatmap(dom_summary, fig_dir / "dense_spine_distribution.png")
    log.info("Wrote dense_spine_distribution.png")

    render_focus_levels(df, fig_dir / "dense_spine_focus.png")
    log.info("Wrote dense_spine_focus.png")


if __name__ == "__main__":
    main()
