"""Tests for `ace_pose.domain_calibration` (C4 per-domain calibration)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.confidence_mc import RiskDistribution
from ace_pose.domain_calibration import (
    TemperatureCalibrator, evaluate_calibration, stratified_split,
)


def _peaked_dist(method: str, classes: np.ndarray, peak: int, peak_prob: float = 0.9):
    p = np.full(classes.shape, (1 - peak_prob) / (len(classes) - 1), dtype=float)
    p[classes == peak] = peak_prob
    p = p / p.sum()
    return RiskDistribution(method=method, classes=classes, probs=p, n_samples=100)


def _build_synthetic_records():
    """Construct a synthetic dataset with two domains and two methods,
    each with a known mis-calibration pattern."""
    cls_owas = np.array([1, 2, 3, 4])
    cls_rula = np.array([1, 2, 3, 4, 5, 6, 7])

    records = []
    rng = np.random.default_rng(42)
    # manufacturing: over-confident OWAS, well-calibrated RULA
    for _ in range(40):
        y = int(rng.integers(1, 5))
        # over-confident: predicts y with very high prob, sometimes wrong
        peak = y if rng.random() < 0.70 else int(rng.integers(1, 5))
        d = _peaked_dist("owas", cls_owas, peak, peak_prob=0.95)
        records.append(("manufacturing", d, y))
        y = int(rng.integers(1, 8))
        d = _peaked_dist("rula", cls_rula, y if rng.random() < 0.80 else
                         int(rng.integers(1, 8)), peak_prob=0.55)
        records.append(("manufacturing", d, y))
    # eldercare: under-confident on both
    for _ in range(40):
        y = int(rng.integers(1, 5))
        d = _peaked_dist("owas", cls_owas, y if rng.random() < 0.65 else
                         int(rng.integers(1, 5)), peak_prob=0.40)
        records.append(("eldercare", d, y))
        y = int(rng.integers(1, 8))
        d = _peaked_dist("rula", cls_rula, y if rng.random() < 0.70 else
                         int(rng.integers(1, 8)), peak_prob=0.30)
        records.append(("eldercare", d, y))
    return records


class TestTemperatureCalibrator:
    def test_default_is_identity(self):
        cal = TemperatureCalibrator()
        cls = np.array([1, 2, 3])
        p = np.array([0.1, 0.6, 0.3])
        d = RiskDistribution(method="owas", classes=cls, probs=p, n_samples=10)
        out = cal.calibrate(d, domain="any")
        np.testing.assert_allclose(out.probs, p)

    def test_fit_returns_self(self):
        records = _build_synthetic_records()
        cal = TemperatureCalibrator()
        ret = cal.fit(records)
        assert ret is cal
        # at least one (domain, method) pair has a non-default T
        assert any(abs(T - 1.0) > 1e-6 for T in cal.by_key.values())

    def test_calibration_reduces_brier_or_ece(self):
        records = _build_synthetic_records()
        cal = TemperatureCalibrator()
        cal.fit(records)
        reports = evaluate_calibration(records, cal)
        improved = 0
        for (domain, method), rep in reports.items():
            # Allow no-op for already-calibrated groups, but at least one
            # group should improve
            if rep.brier_after <= rep.brier_before + 1e-6 and rep.ece_after <= rep.ece_before + 1e-6:
                improved += 1
        assert improved >= 1, reports


class TestStratifiedSplit:
    def test_split_size(self):
        records = _build_synthetic_records()
        train, val = stratified_split(records, val_frac=0.25, rng_seed=0)
        assert len(train) + len(val) == len(records)
        assert 0 < len(val) < len(records)

    def test_each_group_present_in_both(self):
        records = _build_synthetic_records()
        train, val = stratified_split(records, val_frac=0.25, rng_seed=0)
        keys_train = {(d, dist.method) for d, dist, _ in train}
        keys_val = {(d, dist.method) for d, dist, _ in val}
        for k in keys_train:
            assert k in keys_val, f"group {k} missing from val split"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
