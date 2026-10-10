# PWW-DISCLOSURE: public
"""Copy the figures that the cohort scripts write under ace_pose/_runs/batch_v1/analysis/figures
into topic/E/paper/figures under the names the manuscript uses.

Figures whose source was not regenerated in this run (e.g. image-based ones when
the images are not available) are left as they are and listed.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "ace_pose/_runs/batch_v1/analysis/figures"
DST = ROOT / "topic/E/paper/figures"
MAP = {
    "pose_overview.png": "fig1_worker_pose_overlay.png",           # Fig. 3
    "pose_load_overview.png": "fig2_per_environment_load.png",     # Fig. 4
    "score_distribution_by_domain.png": "fig4_score_distribution.png",  # Fig. 6
    "reliability_diagram.png": "fig6_reliability_diagram.png",     # Fig. 8
    "model_accuracy_comparison.png": "supplementary/S2.png",
    "pose_overlay_comparison.png": "supplementary/S3.png",
    "pose_models_grid.png": "supplementary/S4.png",            # fig_pose_models_grid.py (GPU, images)
    "c4_calibration_improvement.png": "supplementary/S5.png",
    "confidence_vs_score.png": "supplementary/S6.png",
    "risk_heatmap_location_x_process.png": "supplementary/S7.png",
    "risk_heatmap_location_x_predprocess.png": "supplementary/S8.png",
}

missing = []
for src, dst in MAP.items():
    if (SRC / src).is_file():
        (DST / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / src, DST / dst)
        print(f"copied {src} -> {dst}")
    else:
        missing.append(src)
for m in missing:
    print(f"not regenerated in this run (left unchanged): {m}")
sys.exit(0)
