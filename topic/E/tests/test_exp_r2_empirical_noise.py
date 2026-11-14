# PWW-DISCLOSURE: public
"""Unit tests for the deterministic helpers of exp_r2_empirical_noise (run with pytest, conda env bioera_mirror_test)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_action_agreement as EA  # noqa: E402
import exp_r2_empirical_noise as E  # noqa: E402
from ace_pose.confidence_mc import apply_temperature  # noqa: E402
from ace_pose.major_joints import NECK, PELVIS  # noqa: E402


def _pose(rng, scale=1.0, shift=0.0):
    kp = rng.normal(size=(25, 3)) * scale + shift
    kp[NECK] = (kp[5] + kp[6]) / 2
    kp[PELVIS] = (kp[11] + kp[12]) / 2
    return kp


def _shape(kp):
    """pelvis origin, own torso length -> comparable across coordinate frames."""
    return (kp[E.J, :2] - kp[PELVIS, :2]) / np.linalg.norm(kp[NECK, :2] - kp[PELVIS, :2])


def test_residual_round_trip_visible_joints():
    rng = np.random.default_rng(1)
    pred, ann = _pose(rng, 3.0, 100.0), _pose(rng, 50.0, -7.0)      # different frames and scales
    conf = np.ones(25)
    conf[[0, 1]] = 0                                                # two joints not annotated
    res = E.residual(pred, ann, conf)
    assert (res[[0, 1]] == 0).all() if E.J[0] == 0 else True
    out = E.apply_residual(pred, res)
    vis = conf[E.J] > 0
    np.testing.assert_allclose(_shape(out)[vis], _shape(ann)[vis], atol=1e-9)


def test_residual_none_when_torso_unusable():
    rng = np.random.default_rng(2)
    pred, ann = _pose(rng), _pose(rng)
    conf = np.ones(25)
    conf[NECK] = 0
    assert E.residual(pred, ann, conf) is None


def test_zero_residual_is_identity_for_midpoint_poses():
    pred = _pose(np.random.default_rng(3))
    np.testing.assert_allclose(E.apply_residual(pred, np.zeros((len(E.J), 2))), pred)


def test_contiguous_set_construction():
    p = np.array([0.05, 0.10, 0.50, 0.30, 0.05])
    assert E.set_for_tau(p, 0.4) == (2, 2)          # mode alone
    assert E.set_for_tau(p, 0.7) == (2, 3)          # larger neighbour (0.30 > 0.10) first
    assert E.set_for_tau(p, 0.85) == (1, 3)
    assert E.set_for_tau(p, 5.0) == (0, 4)          # beyond path -> whole range
    assert E.set_for_tau(np.array([0.5, 0.5]), 0.9) == (0, 1)
    assert E.set_for_tau(np.array([0.3, 0.4, 0.3]), 0.5) == (0, 1)   # tie -> lower side
    # score of a class = u of the first nested set containing it; the set at tau = score covers it
    for y in range(5):
        assert (lambda lh: lh[0] <= y <= lh[1])(E.set_for_tau(p, E.score_of(p, y)))
    # zero-probability class is still reachable
    q = np.array([1.0, 0.0, 0.0])
    lo, hi = E.set_for_tau(q, E.score_of(q, 2))
    assert lo <= 2 <= hi


def test_finite_sample_conformal_quantile():
    s = np.arange(1, 11)                              # n = 10 -> rank ceil(11*0.9) = 10
    assert E.conformal_quantile(s, 0.10) == 10
    assert np.isinf(E.conformal_quantile(s, 0.05))    # rank ceil(11*.95)=11 > n
    s = np.arange(1, 100)                             # n = 99 -> rank ceil(100*.9) = 90
    assert E.conformal_quantile(s[::-1], 0.10) == 90


def test_helpers_match_library():
    rng = np.random.default_rng(4)
    P = rng.dirichlet(np.ones(5), 50)
    np.testing.assert_allclose(E.softmax_T(P, 2.5)[3], apply_temperature(P[3], 2.5))
    from ace_pose.confidence_mc import RiskDistribution, expected_calibration_error
    y = rng.integers(0, 5, 50)
    d = [RiskDistribution("x", np.arange(5), p, 0) for p in P]
    assert abs(E.ece_vec(P, y) - expected_calibration_error(d, list(y), 10)) < 1e-12


def test_no_leakage_donors_disjoint_from_evaluated_workers():
    """Every fold: donor workers and evaluated workers are disjoint (same construction as build_folds)."""
    wk = np.array([f"f_{i % 7}" for i in range(70)])
    E.G["WK"], E.G["VALID"], E.G["MC"] = wk, np.ones(70, bool), np.random.default_rng(0).random(70)
    import pandas as pd
    coh = pd.DataFrame({"domain": ["eldercare"] * 70, "factory_id": ["f"] * 70, "worker_id": [i % 7 for i in range(70)]})
    class _E12:
        SEED, N_MANUFACTURING_REPEATS = 0, 0

    folds, fw = E.build_folds(coh, _E12)
    assert len(folds) == 7
    for f in folds:
        assert not (set(wk[f["donors"]]) & set(wk[f["val"]]))
        assert not f["isval"][f["donors"]].any()
    ev = fw[fw.role == "evaluated"].groupby("fold").worker.apply(set)
    dn = fw[fw.role == "train_donor"].groupby("fold").worker.apply(set)
    assert all(not (ev[k] & dn[k]) for k in ev.index)


# ---------------------------------------------------------------- exploratory additions (reviewer-driven)
def test_ece_cc_and_brier_rows_consistent():
    """A1/A4: ece_cc hand value (0.95 bin: 2/3*|0.5-0.95|, 0.55 bin: 1/3*|1-0.55|) and brier_vec == mean of brier_rows."""
    assert abs(E.ece_cc(np.array([0.95, 0.95, 0.55]), np.array([1.0, 0.0, 1.0])) - 0.45) < 1e-12
    P, y = np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([0, 0])
    assert np.allclose(E.brier_rows(P, y), [0.09 + 0.09, 0.64 + 0.64])     # (1-p)^2 + p_other^2
    assert E.brier_vec(P, y) == E.brier_rows(P, y).mean()


def test_wk_boot_resamples_workers_and_is_deterministic():
    """A1: two workers with values 0 and 1 -> mean in [0, 1] and the extremes are reached; same seed, same interval."""
    wk = np.array(["a"] * 5 + ["b"] * 5)
    v = np.array([0.0] * 5 + [1.0] * 5)
    pt, lo, hi = E.wk_boot(wk, lambda ix: v[ix].mean(), n_boot=300)
    assert pt[0] == 0.5 and lo[0] == 0.0 and hi[0] == 1.0
    again = E.wk_boot(wk, lambda ix: v[ix].mean(), n_boot=300)
    assert np.array_equal(again[1], lo) and np.array_equal(again[2], hi)
    c = np.ones(10)
    assert E.wk_boot(wk, lambda ix: c[ix].mean(), n_boot=50)[1][0] == 1.0     # constant statistic -> degenerate interval


def test_mean_by_key_and_explor_block_small_domains_get_no_interval():
    """A1: an image seen in two folds is averaged for 'image_once'; eldercare gets NaN interval + per-fold values, manufacturing gets one."""
    u, (m,) = E.mean_by_key(np.array([3, 5, 3]), np.array([1.0, 4.0, 3.0]))
    assert list(u) == [3, 5] and list(m) == [2.0, 4.0]
    wk_all = np.array(["w1", "w1", "w2", "w3"])
    vals = [np.array([0, 1]), np.array([2, 3])]
    arrs = dict(x=[np.array([0.0, 1.0]), np.array([1.0, 1.0])])
    stat = lambda x: np.array([x.mean()])  # noqa: E731
    b, w = E.explor_block(dict(analysis="t"), "eldercare", vals, wk_all, arrs, stat, ["mean"])
    assert all(np.isnan(r["ci_lo"]) for r in b) and len(w) == 2 and w[0]["worker"] == "w1" and w[1]["mean"] == 1.0
    b, w = E.explor_block(dict(analysis="t"), "manufacturing", vals, wk_all, arrs, stat, ["mean"])
    assert not np.isnan(b[0]["ci_lo"]) and w == [] and b[0]["n_workers"] == 3 and b[0]["n_images"] == 4


def test_ab_stat_identical_distributions_zero_difference_and_sigma_default():
    """A4: comparing A1 with itself gives zero differences; the sigma sweep's 0.04 / seed 0 config equals the main Gaussian config."""
    from ace_pose.confidence_mc import McConfig
    P, y = np.array([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]]), np.array([0, 1, 1])
    v = E.ab_stat(P, P, y)
    assert v[2] == 0 and v[5] == 0 and v[0] == v[1] and v[3] == v[4]
    assert McConfig(n_samples=E.X.MC_CFG.n_samples, sigma_fraction=E.GAUSS_SIGMAS[0], rng_seed=0) == E.X.MC_CFG


def test_explor_seed_zero_is_main():
    """A5: seed repetition always includes seed 0 (the main run), whatever EXPLOR_SEEDS says."""
    assert 0 in E.X.EXPLOR_SEEDS
