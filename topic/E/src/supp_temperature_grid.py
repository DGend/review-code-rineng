# PWW-DISCLOSURE: public
"""Sensitivity of Table 2 to the temperature search grid.

The main analysis fits T by NLL over the pre-specified grid
[0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5] (ace_pose.confidence_mc.fit_temperature),
and cells that select a grid bound (T = 0.5 or T = 5) may owe their residual
ECE to the grid. This refits on the same split
(ace_pose.run_c4_calibration: stratified 80/20, seed 0) with the grid extended
down to T = 0.1 and up to T = 100 and reports, per (domain, method), the fitted T and validation ECE
under both grids. Table 2 itself keeps the pre-specified grid.

Source data: ace_pose/_runs/batch_v1/summary_with_mc.csv, mc_distributions.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.domain_calibration import TemperatureCalibrator, evaluate_calibration, stratified_split
from ace_pose.run_c4_calibration import _build_records

BATCH = ROOT / "ace_pose/_runs/batch_v1"
DEFAULT = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
EXTENDED = np.concatenate([[0.1, 0.2, 0.3, 0.4], DEFAULT, [7.5, 10, 15, 20, 30, 50, 100]])
THRESH = {"owas": 0.05, "rula": 0.07, "reba": 0.10}


def main():
    df = pd.read_csv(BATCH / "summary_with_mc.csv")
    npz = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
    train, val = stratified_split(_build_records(df, npz), val_frac=0.2, rng_seed=0)
    res = {}
    for name, grid in (("default", DEFAULT), ("extended", EXTENDED)):
        cal = TemperatureCalibrator().fit(train, grid=grid)
        rep = evaluate_calibration(val, cal)
        res[name] = {k: (cal.get_T(*k), r.ece_before, r.ece_after) for k, r in rep.items()}
    print("| Domain | Method | T (grid 0.5-5) | ECE after | T (grid 0.1-100) | ECE after | Threshold |")
    for k in sorted(res["default"]):
        td, b, ed = res["default"][k]
        te, _, ee = res["extended"][k]
        flag = "pass" if ee <= THRESH[k[1]] else "fail"
        print(f"| {k[0]} | {k[1].upper()} | {td:g} | {ed:.3f} | {te:g} | {ee:.3f} | {THRESH[k[1]]} ({flag}) |")


if __name__ == "__main__":
    main()
