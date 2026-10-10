# PWW-DISCLOSURE: public
"""Figure 7: compression-to-tolerance band distribution across the seven
demographic reference profiles (tolerance-only sensitivity analysis, fixed
70 kg body mass; Section 4.6, Table 3).

Why regenerated: the previous figure (copied from topic/D) used the band
names "Safe" and "Disc-failure", which the manuscript replaced on the
ergonomist's advice, labelled the y-axis as a share of poses although the
values are shares of lumbar-disc cells, and hard-coded its numbers. This reads
them from the analysis output instead.

Source data: ace_pose/_runs/batch_v1/analysis/cohort_disc_risk_by_profile.csv
(scope ALL, 24,028 lumbar-disc cells per profile).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "ace_pose/_runs/batch_v1/analysis/cohort_disc_risk_by_profile.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "paper/figures"
BANDS = [("safe", "Within reference range (<33%)", "#2e8b57"),
         ("caution", "Moderate ratio (33-66%)", "#f0c419"),
         ("elevated", "Approaching tolerance (66-100%)", "#e67e22"),
         ("surgical", "Exceeds estimated tolerance (>100%)", "#c0392b")]
ORDER = ["25 M", "40 M", "55 M", "25 F", "40 F", "55 F", "65 F"]


def main():
    df = pd.read_csv(CSV)
    df = df[df.scope == "ALL"].set_index("profile").loc[ORDER]
    print(df[[b[0] for b in BANDS]].round(1).to_string())

    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bottom = np.zeros(len(ORDER))
    for col, label, color in BANDS:
        ax.bar(x, df[col].values, bottom=bottom, label=label, color=color)
        bottom += df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel(f"Share of lumbar-disc cells (%, n = {int(df.n_cells.iloc[0]):,} per profile)")
    ax.set_xlabel("Reference demographic profile (age, sex), fixed 70 kg body mass")
    ax.set_title("Same postures, different estimated tolerance: compression-to-tolerance ratio bands\n"
                 "(body-weight-induced compression only; bands are an analysis convention)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig9_demographic_disc.png", dpi=200)
    fig.savefig(OUT_DIR / "fig9_demographic_disc.pdf")
    print("wrote", OUT_DIR / "fig9_demographic_disc.png")


if __name__ == "__main__":
    main()
