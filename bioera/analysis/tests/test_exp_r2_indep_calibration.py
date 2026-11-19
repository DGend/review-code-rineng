# PWW-DISCLOSURE: public
"""Deterministic helpers of exp_r2_indep_calibration: the 5-95 interval and the reliability bins."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_r2_indep_calibration import interval_5_95, reliability_bins  # noqa: E402


def test_interval_5_95():
    assert interval_5_95(np.array([0.0, 1.0, 0.0, 0.0]), 1) == (2, 2)
    assert interval_5_95(np.array([0.5, 0.4, 0.1]), 1) == (1, 3)   # cdf 0.5, 0.9, 1.0 -> 95th at class 3


def test_reliability_bins():
    b = reliability_bins(np.array([0.95, 0.95, 0.55]), np.array([1.0, 0.0, 1.0]))
    assert b == [(0.55, 1.0, 1), (0.95, 0.5, 2)]


def test_explor_seed_env_default():
    """A5: EXPLOR_SEEDS defaults to seeds 0, 1, 2 (the explor worker-bootstrap of this script is tested through exp_r2_empirical_noise)."""
    import os
    from exp_r2_indep_calibration import EXPLOR_SEEDS
    assert EXPLOR_SEEDS == [int(x) for x in os.environ.get("EXPLOR_SEEDS", "0,1,2").split(",") if x]
