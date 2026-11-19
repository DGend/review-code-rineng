# PWW-DISCLOSURE: public
"""Figure 10: cross-method agreement (OWAS/RULA/REBA), regenerated.

Why regenerated: the previous fig8_cross_method_agreement.png could not
be reproduced from any script in the repo, and its annotated values
(e.g. RULA-REBA Spearman rho = 0.85) did not match Table 5 (rho =
0.548), computed live below from ace_pose.analysis.ErgoAnalyzer against
the current batch. Table 5's numbers ARE reproducible from this script
(see printed values vs manuscript/04.results.md Table 5) -- this
figure now reports exactly those same three statistics, computed the
same way, so the figure and table cannot drift apart again.

Source data (real, not fabricated):
  ace_pose/_runs/batch_v1/summary_with_xw.csv (n=6,007, status==ok)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.analysis import ErgoAnalyzer

OUT_DIR = Path(__file__).resolve().parents[1] / "paper/figures"
BATCH_CSV = ROOT / "ace_pose/_runs/batch_v1/summary_with_xw.csv"

DOMAIN_COLOR = {"manufacturing": "#4472c4", "logistics": "#2e8b57", "eldercare": "#e67e22"}
DOMAIN_LABEL = {"manufacturing": "Manufacturing", "logistics": "Logistics", "eldercare": "Eldercare"}


def main():
    df = pd.read_csv(BATCH_CSV)
    df = df[df["status"] == "ok"].reset_index(drop=True)

    analyzer = ErgoAnalyzer(df)
    agr = analyzer.cross_method_agreement().pairs.set_index(["method_a", "method_b"])
    print(agr)  # sanity check against manuscript Table 5 at generation time

    methods = ["owas", "rula", "reba"]
    labels = {"owas": "OWAS", "rula": "RULA", "reba": "REBA"}

    def get(ma, mb, col):
        if (ma, mb) in agr.index:
            return agr.loc[(ma, mb), col]
        return agr.loc[(mb, ma), col]

    fig = plt.figure(figsize=(13.5, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # ---- panel (a): heatmap, upper = quadratic kappa, lower = spearman rho
    n = len(methods)
    grid = np.full((n, n), np.nan)
    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            if i == j:
                continue
            if j > i:
                grid[i, j] = get(ma, mb, "kappa_quadratic")
            else:
                grid[i, j] = get(mb, ma, "rho")

    cmap = LinearSegmentedColormap.from_list("agree", ["#c0392b", "#f5f0a1", "#2e8b57"])
    im = ax1.imshow(grid, cmap=cmap, vmin=0.0, vmax=0.6)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax1.text(j, i, labels[methods[i]], ha="center", va="center",
                         fontweight="bold", fontsize=13)
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee", zorder=0))
                continue
            val = grid[i, j]
            color = "white" if val > 0.4 else "black"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                      fontweight="bold", fontsize=13, color=color)
            if j > i:
                lk = get(methods[i], methods[j], "kappa_linear")
                ax1.text(j, i + 0.22, f"(linear κ = {lk:.3f})", ha="center", va="center",
                          fontsize=8, color=color, style="italic")

    ax1.set_xticks(range(n)); ax1.set_xticklabels([labels[m] for m in methods])
    ax1.set_yticks(range(n)); ax1.set_yticklabels([labels[m] for m in methods])
    ax1.set_title(f"(a) Pairwise cross-method agreement\n(n = {len(df):,} poses)", fontweight="bold", pad=14)
    ax1.text(1.28, 0.82, "Upper:\nquadratic κ\n(grand score)", ha="center", va="center", fontsize=8.5,
              style="italic", transform=ax1.transAxes,
              bbox=dict(boxstyle="round", facecolor="#f2f2f2", edgecolor="none"))
    ax1.text(-0.30, 0.22, "Lower:\nSpearman ρ\n(grand score)", ha="center", va="center", fontsize=8.5,
              style="italic", transform=ax1.transAxes,
              bbox=dict(boxstyle="round", facecolor="#f2f2f2", edgecolor="none"))
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", pad=0.18, fraction=0.05)
    cbar.set_label("Agreement (κ or ρ)")

    # ---- panel (b): RULA vs REBA scatter, coloured by domain
    # both scores are small integers, so exact overlaps are common; jitter
    # for visibility and draw smallest groups last so they aren't hidden
    # under the majority (manufacturing) class.
    rho = get("rula", "reba", "rho")
    kq = get("rula", "reba", "kappa_quadratic")
    kl = get("rula", "reba", "kappa_linear")
    rng = np.random.default_rng(0)
    domain_order = df["domain"].value_counts().sort_values(ascending=False).index.tolist()
    for dom in domain_order:
        sub = df[df["domain"] == dom]
        jx = sub["rula_grand"] + rng.uniform(-0.15, 0.15, len(sub))
        jy = sub["reba_grand"] + rng.uniform(-0.15, 0.15, len(sub))
        ax2.scatter(jx, jy, s=14, alpha=0.55,
                    color=DOMAIN_COLOR.get(dom, "#999999"), label=f"{DOMAIN_LABEL.get(dom, dom)} (n={len(sub)})")
    lo, hi = df["rula_grand"].min(), df["rula_grand"].max()
    ax2.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x reference")
    b, a = np.polyfit(df["rula_grand"], df["reba_grand"], 1)
    xs = np.linspace(lo, hi, 50)
    ax2.plot(xs, a + b * xs, color="#c0392b", linewidth=2, label="Linear fit")
    ax2.set_xlabel("RULA grand score"); ax2.set_ylabel("REBA grand score")
    ax2.set_title(f"(b) RULA vs REBA grand score\n(by domain, n = {len(df):,} poses)", fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.text(0.03, 0.97,
              f"Spearman ρ = {rho:.3f}\nQuadratic κ = {kq:.3f}\nLinear κ = {kl:.3f}",
              transform=ax2.transAxes, va="top", fontsize=11, fontweight="bold",
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="#c0392b"))

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig8_cross_method_agreement.png", dpi=200)
    print("wrote", OUT_DIR / "fig8_cross_method_agreement.png")


if __name__ == "__main__":
    main()
