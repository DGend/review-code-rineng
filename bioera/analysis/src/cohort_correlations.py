# PWW-DISCLOSURE: public
"""Section 4.6 and Supplementary Fig. S8 cohort statistics.

Why this exists: the manuscript's earlier values (L5/S1-REBA Spearman rho 0.46;
S8 predicted-vs-ground-truth ranking rho 0.91) had no producer script and are
not reproduced by the cohort data. This prints the values the text now uses.

Source data: ace_pose/_runs/batch_v1/summary_with_xw.csv (n = 6,007)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "ace_pose/_runs/batch_v1/summary_with_xw.csv"


def main():
    d = pd.read_csv(CSV)
    d = d[d.status == "ok"]
    rho, p = spearmanr(d.biomech_L5S1_Nm, d.reba_grand)
    print(f"L5/S1 moment vs REBA grand: Spearman rho = {rho:.2f}, p = {p:.1e}, n = {len(d)}")
    med = d.groupby("domain").biomech_L5S1_Nm.median()
    print("median L5/S1 moment by domain (Nm):", med.round(1).to_dict())
    e = d.biomech_L5S1_Nm[d.domain == "eldercare"].to_numpy()
    m = d.biomech_L5S1_Nm[d.domain == "manufacturing"].to_numpy()
    p_mw = mannwhitneyu(e, m, alternative="greater").pvalue
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(e, len(e))) / np.median(rng.choice(m, len(m))) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"eldercare/manufacturing median L5/S1 ratio = {np.median(e) / np.median(m):.2f}, "
          f"bootstrap 95% CI [{lo:.2f}, {hi:.2f}], one-sided Mann-Whitney p = {p_mw:.2g}")

    gt = d.groupby("process_id").agg(n=("reba_grand", "size"), m=("reba_grand", "mean"))
    gt = gt[gt.n >= 30].sort_values("m", ascending=False).head(15)
    pred = d.groupby("pred_process_id").reba_grand.mean()
    common = [i for i in gt.index if i in pred.index]
    rho_s8, _ = spearmanr(gt.loc[common, "m"], pred.loc[common])
    print(f"S8: top-15 ground-truth vs predicted-process mean REBA, Spearman rho = {rho_s8:.2f} "
          f"(n = {len(common)}); process-classifier accuracy {d.pred_process_correct.mean():.1%}")


if __name__ == "__main__":
    main()
