"""Generate manuscript-ready figures from batch outputs.

Plots produced (saved to `batch/analysis/figures/`):

  1. risk_heatmap_location_x_process.png
     RULA mean grid: rows = factory location, cols = process_label.
  2. risk_heatmap_location_x_predprocess.png
     Same grid using Y predicted process IDs (showcases Y's utility).
  3. score_distribution_by_domain.png
     Histogram overlay of RULA / REBA grand scores per domain.
  4. top_processes.png
     Bar chart: top-15 highest-risk processes, ranked by mean REBA
     grand score (RULA mean + REBA mean shown per process).
  5. reliability_diagram.png
     C3 calibration reliability diagram per method × domain (one panel
     per method, three coloured lines).
  6. confidence_vs_score.png
     Scatter: keypoint mean confidence vs MC RULA expected score —
     visualises the noisy-pose-inflates-risk effect.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.analysis import FACTORY_LABEL
from ace_pose.batch_infer import PROCESS_LABEL

# Romanised process labels for matplotlib (Korean glyphs are missing
# from DejaVu Sans). Used for figures only; CSV outputs keep the
# native Korean labels.
PROCESS_LABEL_EN = {
    1: "Sasang-1", 2: "Sasang-2", 3: "Sasang-3", 4: "Sasang-4",
    5: "Sasang-5", 6: "Sasang-6",
    7: "Weld-1", 8: "Weld-2", 9: "Weld-3", 10: "Weld-4",
    11: "Cast", 12: "Bend", 13: "Cut", 14: "Assemble", 15: "Process",
    16: "Sort", 17: "Paint", 18: "Move", 19: "Assist",
    20: "Bathing", 21: "Wash", 22: "Massage", 23: "Meal-aid",
}


def _romanise_process(label):
    """Map a process_label (string) or process_id (int) to a Roman label."""
    # if it's an integer-like (predicted process), use the int map
    try:
        i = int(label)
        return PROCESS_LABEL_EN.get(i, str(label))
    except (ValueError, TypeError):
        pass
    # otherwise it's a Korean string — invert PROCESS_LABEL to find ID
    for pid, kor in PROCESS_LABEL.items():
        if kor == label:
            return PROCESS_LABEL_EN.get(pid, label)
    return str(label)

log = logging.getLogger("ace_pose.run_plots")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300         # CBM minimum
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["pdf.fonttype"] = 42         # embed TrueType fonts (editable PDF)
plt.rcParams["ps.fonttype"] = 42


def _save_fig(fig, out_path: Path, dpi: int = 300):
    """Save a matplotlib figure to PNG (300 dpi) and PDF (vector) variants."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi)                # PNG
    fig.savefig(out_path.with_suffix(".pdf"))     # vector PDF
    log.info("Wrote %s + .pdf", out_path)


def _heatmap(ax, matrix, xticks, yticks, title, vmin=None, vmax=None, cbar_label=""):
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label(cbar_label)
    # value annotations for small grids
    if matrix.size <= 100:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > (vmax or matrix.max()) * 0.6 else "black",
                        fontsize=8)


def plot_heatmap_location_x_process(df: pd.DataFrame, out_path: Path,
                                    proc_col: str = "process_label",
                                    title_suffix: str = ""):
    pivot = df.groupby(["location", proc_col])["rula_grand"].mean().unstack(fill_value=np.nan)
    xticks = [_romanise_process(c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(min(0.7 * pivot.shape[1] + 2, 16), 0.5 * pivot.shape[0] + 2))
    _heatmap(
        ax, pivot.values, xticks, list(pivot.index),
        f"Mean RULA grand by location × {proc_col}{title_suffix}",
        vmin=1, vmax=7, cbar_label="RULA grand",
    )
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_score_distribution_by_domain(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    methods = [("owas_action", "OWAS action category", (0.5, 4.5)),
               ("rula_grand", "RULA grand score", (0.5, 7.5)),
               ("reba_grand", "REBA grand score", (0.5, 15.5))]
    domains = sorted(df["domain"].dropna().unique())
    for ax, (col, label, xlim) in zip(axes, methods):
        for dom in domains:
            sub = df.loc[df["domain"] == dom, col].dropna()
            ax.hist(sub, bins=np.arange(xlim[0], xlim[1] + 1, 1) - 0.5,
                    alpha=0.5, label=f"{dom} (n={len(sub)})", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_xlim(xlim)
        ax.legend(fontsize=8)
    fig.suptitle("Score distribution by domain")
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_top_processes(df: pd.DataFrame, out_path: Path,
                       n_top: int = 15, min_n: int = 30):
    grouped = df.groupby("process_label").agg(
        rula_mean=("rula_grand", "mean"),
        reba_mean=("reba_grand", "mean"),
        n=("rula_grand", "size"),
    )
    grouped = grouped.loc[grouped["n"] >= min_n].sort_values("reba_mean", ascending=False).head(n_top)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(grouped))
    width = 0.4
    ax.barh(y - width / 2, grouped["rula_mean"], width, label="RULA", color="#2563eb")
    ax.barh(y + width / 2, grouped["reba_mean"], width, label="REBA", color="#dc2626")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{_romanise_process(p)} (n={n})" for p, n in zip(grouped.index, grouped["n"])
    ])
    ax.invert_yaxis()
    ax.set_xlabel("Mean grand score")
    ax.set_title(f"Top-{n_top} highest-risk processes (n≥{min_n})")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_reliability_diagram(df: pd.DataFrame, npz_data, out_path: Path,
                             n_bins: int = 10):
    """Per-method reliability diagram: x = top-class confidence, y =
    empirical accuracy at that confidence level (using deterministic
    score as the "label"). One subplot per method, lines per domain."""
    method_info = {
        "owas": ("owas_probs", "owas_action", (1, 4)),
        "rula": ("rula_probs", "rula_grand", (1, 7)),
        "reba": ("reba_probs", "reba_grand", (1, 15)),
    }
    file_stems = npz_data["file_stem"]
    stem_idx = {s: i for i, s in enumerate(file_stems)}
    domains = sorted(df["domain"].dropna().unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for ax, (method, (key, label_col, rng)) in zip(axes, method_info.items()):
        probs_full = npz_data[key]
        for dom in domains:
            sub = df.loc[df["domain"] == dom]
            confs = []
            corrects = []
            for _, row in sub.iterrows():
                i = stem_idx.get(row["file_stem"])
                if i is None: continue
                p = probs_full[i]
                if p.sum() < 1e-6: continue
                p = p / p.sum()
                top_idx = int(np.argmax(p))
                top_cls = rng[0] + top_idx
                confs.append(float(p[top_idx]))
                corrects.append(1.0 if top_cls == int(row[label_col]) else 0.0)
            confs = np.array(confs)
            corrects = np.array(corrects)
            if confs.size == 0:
                continue
            mid = []
            acc = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                m = (confs > lo) & (confs <= hi if hi < 1.0 else confs <= hi + 1e-9)
                if m.sum() < 5:
                    continue
                mid.append((lo + hi) / 2)
                acc.append(corrects[m].mean())
            ax.plot(mid, acc, marker="o", label=f"{dom} (n={len(confs)})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.set_xlabel("Top-class confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability: {method.upper()}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams (MC top-class confidence vs deterministic label)")
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_confidence_vs_score(df: pd.DataFrame, out_path: Path):
    if "mc_rula_expected" not in df.columns:
        log.warning("No mc_rula_expected column: skipping confidence_vs_score plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    domains = sorted(df["domain"].dropna().unique())
    colours = {"manufacturing": "#2563eb", "eldercare": "#dc2626", "logistics": "#059669"}
    for dom in domains:
        sub = df.loc[df["domain"] == dom].dropna(subset=["kpt_conf_mean", "mc_rula_expected"])
        ax.scatter(sub["kpt_conf_mean"], sub["mc_rula_expected"],
                   s=4, alpha=0.3, color=colours.get(dom, "#6b7280"),
                   label=f"{dom} (n={len(sub)})")
    ax.set_xlabel("Mean keypoint confidence")
    ax.set_ylabel("MC RULA expected score")
    ax.set_title("Pose confidence vs MC-RULA expected risk")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_calibration_improvement(rep_csv: Path, out_path: Path):
    if not rep_csv.exists():
        log.warning("No calibration report at %s: skipping", rep_csv)
        return
    df = pd.read_csv(rep_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["brier", "ece"]):
        ind = np.arange(len(df))
        width = 0.4
        ax.bar(ind - width / 2, df[f"{metric}_before"], width,
               label=f"{metric.upper()} before", color="#9ca3af")
        ax.bar(ind + width / 2, df[f"{metric}_after"], width,
               label=f"{metric.upper()} after", color="#2563eb")
        ax.set_xticks(ind)
        ax.set_xticklabels([f"{r.domain}/{r.method}" for r in df.itertuples()],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"C4 calibration: {metric.upper()} per (domain, method)")
        ax.set_ylabel(metric.upper())
        ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    args = parser.parse_args()
    batch_dir = Path(args.batch)
    fig_dir = batch_dir / "analysis" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    sum_path = batch_dir / "summary_with_mc.csv"
    if not sum_path.exists():
        sum_path = batch_dir / "summary_with_pred.csv"
    if not sum_path.exists():
        sum_path = batch_dir / "summary.csv"
    df = pd.read_csv(sum_path)
    df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")

    log.info("Plotting from %s (%d rows)", sum_path, len(df))

    plot_score_distribution_by_domain(df, fig_dir / "score_distribution_by_domain.png")
    plot_top_processes(df, fig_dir / "top_processes.png")
    plot_heatmap_location_x_process(df, fig_dir / "risk_heatmap_location_x_process.png",
                                    proc_col="process_label",
                                    title_suffix=" (GT process)")
    if "pred_process_id" in df.columns and df["pred_process_id"].notna().any():
        sub = df.dropna(subset=["pred_process_id"]).copy()
        sub["pred_process_label"] = sub["pred_process_id"].astype(int).map(PROCESS_LABEL).fillna("?")
        plot_heatmap_location_x_process(
            sub, fig_dir / "risk_heatmap_location_x_predprocess.png",
            proc_col="pred_process_label",
            title_suffix=" (predicted process)")

    # MC + calibration plots, when those columns exist
    mc_npz_path = batch_dir / "mc_distributions.npz"
    if mc_npz_path.exists():
        npz_data = np.load(mc_npz_path, allow_pickle=True)
        plot_reliability_diagram(df, npz_data, fig_dir / "reliability_diagram.png")
        plot_confidence_vs_score(df, fig_dir / "confidence_vs_score.png")
    plot_calibration_improvement(
        batch_dir / "c4_calibration_report.csv",
        fig_dir / "c4_calibration_improvement.png",
    )


if __name__ == "__main__":
    main()
