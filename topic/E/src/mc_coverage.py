# PWW-DISCLOSURE: public
"""Sections 3.4, 3.5, 4.5: how often the Monte Carlo output agrees with the
deterministic score of the unperturbed pose.

Prints, per method, (a) the share of poses whose 5th-95th percentile interval of
sampled grand scores contains the deterministic score, overall and for poses with
mean keypoint confidence of at least 0.6, and (b) the share whose modal sampled
score equals it. Replaces an earlier unsupported statement that the sampling
noise had been tuned on a held-out subset.

Source data: ace_pose/_runs/batch_v1/summary_with_mc.csv, mc_distributions.npz
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "ace_pose/_runs/batch_v1"
METHODS = {"owas": "owas_action", "rula": "rula_grand", "reba": "reba_grand"}


def main():
    df = pd.read_csv(BATCH / "summary_with_mc.csv").set_index("file_stem")
    z = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
    stems = z["file_stem"]
    conf = df.loc[stems, "kpt_conf_mean"].to_numpy()
    for m, col in METHODS.items():
        probs = z[f"{m}_probs"]
        inside, mode_ok, keep = [], [], []
        for i, p in enumerate(probs):
            if p.sum() < 1e-6:
                continue
            p = p / p.sum()
            classes = np.arange(1, len(p) + 1)
            y = int(df.loc[stems[i], col])
            cdf = np.cumsum(p)
            lo = classes[np.searchsorted(cdf, 0.05)]
            hi = classes[min(np.searchsorted(cdf, 0.95), len(p) - 1)]
            inside.append(lo <= y <= hi)
            mode_ok.append(classes[np.argmax(p)] == y)
            keep.append(conf[i] >= 0.6)
        inside, mode_ok, keep = map(np.array, (inside, mode_ok, keep))
        print(f"{m.upper()}: 5-95% interval covers {100*inside.mean():.1f}% of all poses, "
              f"{100*inside[keep].mean():.1f}% of poses with confidence >= 0.6 (n = {keep.sum()}); "
              f"mode matches {100*mode_ok.mean():.1f}%")


if __name__ == "__main__":
    main()
