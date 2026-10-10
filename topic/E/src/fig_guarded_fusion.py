# PWW-DISCLOSURE: public
"""Figure 3: joint-moment effect of naive vs. guarded pose fusion on the four-pose-per-joint panel.

Replaces the earlier Figure 3 (an internal v1/v2 ablation overlay grid that did not show
what its caption described). Adapted from topic/D/src/fig_guarded_fusion.py.

Source data (real, not fabricated):
- ace_pose/_runs/batch_v1/analysis/v1_vs_v2_clinical_summary.csv (n = 4 poses per joint)

"""
import csv
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "ace_pose/_runs/batch_v1/analysis/v1_vs_v2_clinical_summary.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "paper/figures"

JOINT_ORDER = ["L5/S1", "Hip", "Shoulder", "Knee"]
SERIES = [
    ("v1_mean", "Single backbone (v1)", "#7f7f7f"),
    ("v2-CD-soft_mean", "Naive soft fusion", "#4472c4"),
    ("v2-CD-max_mean", "Naive winner-takes-all", "#e67e22"),
    ("v2-guarded_mean", "Guarded fusion (deployed)", "#2e8b57"),
]


def load():
    rows = {}
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            rows[row["joint_name"]] = row
    return rows


def main():
    rows = load()
    x = np.arange(len(JOINT_ORDER))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (col, label, color) in enumerate(SERIES):
        vals = [float(rows[j][col]) for j in JOINT_ORDER]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_ORDER)
    ax.set_ylabel("Joint moment (Nm)")
    ax.set_title(
        "Guarded vs. naive pose fusion: joint-moment effect\non a four-pose-per-joint panel (n = 4)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_guarded_fusion_moments.png", dpi=200)
    fig.savefig(OUT_DIR / "fig3_guarded_fusion_moments.pdf")
    print("wrote", OUT_DIR / "fig3_guarded_fusion_moments.png")


if __name__ == "__main__":
    main()
