# PWW-DISCLOSURE: public
"""Figure 8: per-axis and per-joint Shapley attribution of the RULA grand score.

Why regenerated: the previous fig7_shapley_attribution.png had no producer
script, showed a "Load / Muscle" axis that ace_pose.xai's RULA value function
does not contain, and its shares could not be reproduced from any version of
the cohort. This script computes everything with ace_pose.xai (the module the
paper describes) over every pose at the highest RULA grand score in the cohort, a
tie-free selection, unlike the old "top twenty" which was ambiguous because
more than twenty poses tie at 7.

Source data (real, not fabricated):
  ace_pose/_runs/batch_v1/summary.csv + details/*.npz (re-scored 2026-10-04)
Prints the numbers used in manuscript Section 4.7.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import ace_pose.xai as X
from ace_pose.ergo_v2.context_policy import cohort_context  # posture only, see context_policy
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.major_joints import Major25Pose

BATCH = ROOT / "ace_pose/_runs/batch_v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "paper/figures"
LABELS = {"upper_arm": "Upper arm", "lower_arm": "Lower arm", "wrist": "Wrist",
          "wrist_twist": "Wrist twist", "neck": "Neck", "trunk": "Trunk", "legs": "Legs"}


def main():
    df = pd.read_csv(BATCH / "summary.csv")
    ok = df[df.status == "ok"]
    g_max = int(ok.rula_grand.max())
    top = ok[ok.rula_grand == g_max]
    axis_rows, joint_rows = [], []
    for stem in top.file_stem:
        d = np.load(BATCH / "details" / f"{stem}.npz")
        pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                           confidence=d["conf_m25"].astype(np.float64),
                           source=np.zeros(25, dtype=np.int8))
        rula = score_all(pose, world_up=np.array([0.0, -1.0, 0.0]), walking=False,
                         **cohort_context(ok.set_index("file_stem").at[stem, "process_label"])).rula
        shap = X.shapley_attribution(X._rula_values(rula), X.RULA_AXES, X._eval_rula)
        total = sum(shap.values())
        assert abs(total - (X._eval_rula(X._rula_values(rula))
                            - X._eval_rula(X._axis_min(X.RULA_AXES)))) < 1e-9  # efficiency
        axis_rows.append({k: v / total for k, v in shap.items()})
        joints = X.project_onto_joints(shap, X.RULA_AXIS_JOINTS)
        joint_rows.append({k: v / total for k, v in joints.items()})

    ax_df = pd.DataFrame(axis_rows)
    jt_df = pd.DataFrame(joint_rows).fillna(0.0)
    n = len(ax_df)
    mean = ax_df.mean()
    half = 1.96 * ax_df.std(ddof=1) / np.sqrt(n)
    order = mean.sort_values(ascending=False).index.tolist()
    print(f"n poses at RULA grand {g_max}: {n}")
    for k in order:
        print(f"  {LABELS[k]:12s} {100*mean[k]:5.1f}%  95% CI +/- {100*half[k]:.1f}")
    print(f"  trunk + neck = {100*(mean['trunk']+mean['neck']):.1f}%")
    jmean = jt_df.mean().sort_values(ascending=False)
    print("top joints:", {k: round(100 * v, 1) for k, v in jmean.head(6).items()})

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    a.bar([LABELS[k] for k in order], 100 * mean[order], yerr=100 * half[order],
          capsize=4, color="#c0392b")
    a.set_ylabel("Share of RULA grand-score Shapley total (%)")
    a.set_title(f"(a) Per-axis Shapley share, all n = {n} poses at RULA {g_max}\n(mean, 95% CI)")
    a.tick_params(axis="x", rotation=30)
    top_j = jmean.head(12)[::-1]
    b.barh([j.replace("_", " ").title() for j in top_j.index], 100 * top_j.values, color="#e67e22")
    b.set_xlabel("Share projected onto joint (%)")
    b.set_title("(b) Major-25 joints, axis shares split evenly\nacross each axis's joints (top 12)")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig7_shapley_attribution.png", dpi=200)
    fig.savefig(OUT_DIR / "fig7_shapley_attribution.pdf")
    print("wrote", OUT_DIR / "fig7_shapley_attribution.png")


if __name__ == "__main__":
    main()
