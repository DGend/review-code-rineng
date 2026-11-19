# PWW-DISCLOSURE: public
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_r2_mc_sensitivity import mode_and_coverage


def test_mode_and_coverage():
    p = np.array([0.02, 0.9, 0.08])           # mode class 2; 5-95 interval = classes 2..3
    assert mode_and_coverage(p, 2) == (1, 1)
    assert mode_and_coverage(p, 3) == (0, 1)
    assert mode_and_coverage(p, 1) == (0, 0)


def test_assert_worker_disjoint_synthetic():
    from exp_r2_mc_sensitivity import assert_worker_disjoint
    wk = np.array(["a", "a", "b", "b", "c"])
    assert_worker_disjoint({"d": [([0, 1, 2, 3], [4]), ([2, 3, 4], [0, 1])]}, wk)
    import pytest
    with pytest.raises(AssertionError):
        assert_worker_disjoint({"d": [([0, 2, 4], [1, 3])]}, wk)      # worker a and b on both sides
    with pytest.raises(AssertionError):
        assert_worker_disjoint({"d": [([0, 1], [])]}, wk)             # empty validation


def test_real_folds_worker_disjoint():
    """The folds the experiment really uses (exp_r2_indep_calibration's) never share a worker."""
    import pytest
    import exp_r2_indep_calibration as X
    from exp_r2_mc_sensitivity import assert_worker_disjoint, worker_key
    try:
        coh = X.build_cohort()
    except FileNotFoundError:
        pytest.skip("AI-Hub annotations not available")
    folds = X.make_folds(coh, X._load_exp012())
    assert_worker_disjoint(folds, worker_key(coh))
    assert sum(len(v) for v in folds.values()) >= 20


def test_top2_gap_and_seed_cells():
    """A6: near-tie gap = top1 - top2 share; A5: seed rows carry the cell values of the run they came from."""
    from exp_r2_mc_sensitivity import seed_cells, top2_gap
    g = top2_gap(np.array([[0.5, 0.45, 0.05], [0.9, 0.1, 0.0]]))
    assert np.allclose(g, [0.05, 0.8]) and (g < 0.1).tolist() == [True, False]
    r = seed_cells(2, {("manufacturing", "OWAS"): {"ece_before": 0.1, "ece_after_default_mean": 0.05}})
    assert r == [dict(seed=2, domain="manufacturing", method="OWAS", ece_before=0.1, ece_after_default_mean=0.05)]
