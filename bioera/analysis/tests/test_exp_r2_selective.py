# PWW-DISCLOSURE: public
"""Unit tests for exp_r2_selective (pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exp_r2_selective as S  # noqa: E402


def test_threshold_smallest_meeting_accuracy():
    conf = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0, 1.0])
    ok = np.array([0, 1, 0, 1, 1, 1, 1, 1], bool)
    assert S.select_threshold(conf, ok, 0.96) == 0.8      # conf>=0.8: 5/5; conf>=0.7: 5/6 < 0.96
    assert S.select_threshold(conf, ok, 0.5) == 0.5       # all: 6/8 = 0.75 >= 0.5


def test_threshold_none_when_unreachable():
    assert S.select_threshold([0.5, 0.9], [False, False]) is None


def test_fold_stats_no_acceptance():
    s = S.fold_stats(None, np.array([0.5, 0.9]), np.array([True, False]))
    assert s["share"] == 0 and s["n_acc"] == 0 and np.isnan(s["acc_acc"]) and s["rej_acc"] == 0.5


def test_fold_workers_disjoint():
    import exp_r2_empirical_noise as E
    coh = E.X.build_cohort().sample(600, random_state=0).reset_index(drop=True)
    G = E.G
    G["WK"] = np.array([f"{a}_{b}" for a, b in zip(coh.factory_id, coh.worker_id)])
    for fl in E.X.make_folds(coh, E.X._load_exp012()).values():
        for tr, va in fl:
            assert not set(G["WK"][tr]) & set(G["WK"][va])


# ---------------------------------------------------------------- exploratory additions (reviewer-driven)
def test_acc_at_share_and_aurc_hand_values():
    """A2: accuracy of the top-q most confident; AURC of conf [.9,.8] with the lower one wrong = 0.5 only at q = 1 of 100 grid points."""
    conf, ok = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 1], bool)
    assert S.acc_at_share(conf, ok, 0.5) == 1.0 and S.acc_at_share(conf, ok, 1.0) == 0.75 and S.acc_at_share(conf, ok, 0.1) == 1.0
    assert abs(S.aurc(np.array([0.9, 0.8]), np.array([True, False])) - 0.005) < 1e-12


def test_error_structure_hand_values():
    """A2: errors off by one level, levels present, majority baselines."""
    x = dict(pva=np.array([1, 2, 4, 1]), yva=np.array([1, 1, 2, 3]), ytr=np.array([1, 1, 2]))
    r = S.error_structure([x])
    assert abs(r["adjacent_error_share"] - 1 / 3) < 1e-12 and r["n_levels_present"] == 3
    assert r["majority_share_heldout"] == 0.5 and r["majority_train_level_acc"] == 0.5 and r["mode_acc"] == 0.25


def test_minacc_sweep_monotone_and_split_verdict():
    """A3: a stricter training-accuracy target never accepts more held-out images; criterion() flags the 7-cell split."""
    import pandas as pd
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1, 400)
    ok = rng.uniform(size=400) < conf                                    # more confident -> more often right
    rs = [dict(f={"g": 0}, ctr=conf, oktr=ok, cva=conf, okva=ok)]
    n_acc = [S.cell_stats(rs, ma, dict(src="A1", domain="d", method="M", target="t"))[0]["n_acc"] for ma in S.MIN_ACC_SWEEP]
    assert n_acc == sorted(n_acc, reverse=True)
    d = pd.DataFrame(dict(**{"pass": [True] * 7 + [False] * 2}, share_fold_mean=0.5, n_folds_no_acceptance=[1] + [0] * 8))
    assert S.criterion(d)["verdict"] == "split"
