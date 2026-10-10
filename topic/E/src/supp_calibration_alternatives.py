# PWW-DISCLOSURE: public
"""Supplementary Table S2.1: temperature scaling vs Platt, isotonic, and Beta
calibration on Manufacturing-RULA.

Why this exists: the earlier Table S2.1 reported Platt/isotonic/Beta numbers
that no script in the project produced. This computes all four on exactly the
split and ECE definition of ace_pose.run_c4_calibration (stratified 80/20,
seed 0; top-class confidence, 10 equal-width bins). The three alternatives are
fitted as top-label recalibration maps (raw top-class confidence -> P(correct))
on the training fold and applied to the validation fold; the predicted class
is unchanged. Beta calibration follows Kull et al. 2017: logistic regression
on ln(p) and -ln(1-p).

Source data: ace_pose/_runs/batch_v1/summary_with_mc.csv, mc_distributions.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.confidence_mc import expected_calibration_error
from ace_pose.domain_calibration import TemperatureCalibrator, evaluate_calibration, stratified_split
from ace_pose.run_c4_calibration import _build_records

BATCH = ROOT / "ace_pose/_runs/batch_v1"
DOMAIN, METHOD = "manufacturing", "rula"
EPS = 1e-6


def top_conf_correct(recs):
    c = np.array([float(np.max(d.probs)) for _, d, _ in recs])
    ok = np.array([float(int(d.classes[np.argmax(d.probs)]) == y) for _, d, y in recs])
    return c, ok


def ece(conf, correct, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= (hi if hi < 1.0 else hi + 1e-9))
        if m.any():
            total += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return total


def main():
    df = pd.read_csv(BATCH / "summary_with_mc.csv")
    npz = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
    train, val = stratified_split(_build_records(df, npz), val_frac=0.2, rng_seed=0)
    key = lambda r: r[0] == DOMAIN and r[1].method == METHOD
    tr, va = [r for r in train if key(r)], [r for r in val if key(r)]

    c_tr, y_tr = top_conf_correct(tr)
    c_va, y_va = top_conf_correct(va)
    before = ece(c_va, y_va)
    ref = expected_calibration_error([d for _, d, _ in va], [y for *_, y in va])
    assert abs(before - ref) < 1e-12, (before, ref)  # same ECE as the main paper

    temp = evaluate_calibration(val, TemperatureCalibrator().fit(train))[(DOMAIN, METHOD)]
    logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    platt = LogisticRegression(C=1e6).fit(logit(c_tr)[:, None], y_tr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c_tr, y_tr)
    bfeat = lambda p: np.column_stack([np.log(np.clip(p, EPS, 1)), -np.log(np.clip(1 - p, EPS, 1))])
    beta = LogisticRegression(C=1e6).fit(bfeat(c_tr), y_tr)

    rows = [
        ("Temperature scaling (main paper)", "1 (T)", temp.ece_before, temp.ece_after),
        ("Platt scaling", "2 (a, b)", before, ece(platt.predict_proba(logit(c_va)[:, None])[:, 1], y_va)),
        ("Isotonic regression", "non-parametric", before, ece(iso.predict(c_va), y_va)),
        ("Beta calibration", "3 (a, b, c)", before, ece(beta.predict_proba(bfeat(c_va))[:, 1], y_va)),
    ]
    print(f"n_train={len(tr)}  n_val={len(va)}")
    for name, k, b, a in rows:
        print(f"| {name} | {k} | {b:.3f} | {a:.3f} | {100*(a-b)/b:+.0f}% |")


if __name__ == "__main__":
    main()
