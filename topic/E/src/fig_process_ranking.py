# PWW-DISCLOSURE: public
"""Figure 5: top-15 highest-risk processes, regenerated.

Why regenerated: the checked-in fig5_process_ranking.png was titled
"Top-10 highest-risk processes" and only plotted 10 processes, while
the manuscript text and caption describe fifteen. It also sorted by
mean RULA score while the manuscript text describes ranking by mean
REBA score. Both are fixed at the source (ace_pose.run_plots.
plot_top_processes: n_top=15, sorted by reba_mean) rather than only in
this wrapper, so any future regeneration stays consistent.

Source data (real, not fabricated):
  ace_pose/_runs/batch_v1/summary_with_xw.csv (n=6,007, status==ok)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.run_plots import plot_top_processes

OUT_DIR = Path(__file__).resolve().parents[1] / "paper/figures"
BATCH_CSV = ROOT / "ace_pose/_runs/batch_v1/summary_with_xw.csv"


def main():
    df = pd.read_csv(BATCH_CSV)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_processes(df, OUT_DIR / "fig5_process_ranking.png")
    print("wrote", OUT_DIR / "fig5_process_ranking.png")


if __name__ == "__main__":
    main()
