"""Tests for `ace_pose.confidence_mc` (C3 probabilistic scoring)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.confidence_mc import (
    McConfig, RiskDistribution, apply_temperature, brier_score,
    expected_calibration_error, fit_temperature, mc_score_all, sample_poses,
)
from ace_pose.tests.test_ergo_v2_golden import _upright_skeleton, WORLD_UP


def _high_conf_pose(angles_kwargs: dict | None = None):
    pose = _upright_skeleton(**(angles_kwargs or {}))
    # high confidence everywhere → MC should collapse near deterministic
    pose.confidence[:] = 0.95
    return pose


def _low_conf_pose(angles_kwargs: dict | None = None):
    pose = _upright_skeleton(**(angles_kwargs or {}))
    pose.confidence[:] = 0.20
    return pose


class TestSampling:
    def test_sample_shape_and_perturbation(self):
        pose = _high_conf_pose()
        cfg = McConfig(n_samples=64, sigma_fraction=0.05, rng_seed=42)
        samples = sample_poses(pose, cfg)
        assert len(samples) == 64
        # samples differ from the source
        diffs = np.array([np.linalg.norm(s.keypoints - pose.keypoints) for s in samples])
        assert (diffs > 0).all()

    def test_higher_noise_for_lower_confidence(self):
        cfg = McConfig(n_samples=128, sigma_fraction=0.1, rng_seed=0)
        hi = _high_conf_pose()
        lo = _low_conf_pose()
        hi_samples = sample_poses(hi, cfg)
        lo_samples = sample_poses(lo, cfg)
        hi_var = np.var([s.keypoints for s in hi_samples], axis=0).mean()
        lo_var = np.var([s.keypoints for s in lo_samples], axis=0).mean()
        # low-confidence pose should yield ~9x more variance (using (1-c)² ratio)
        assert lo_var > 5 * hi_var, (hi_var, lo_var)


class TestDistributionalScoring:
    def test_mc_defaults_match_deterministic_scorer(self):
        # The cohort's deterministic scores (score_all defaults) are the
        # calibration labels, so MC must score with the same settings.
        from ace_pose.ergo_v2 import score_all
        pose = _high_conf_pose()
        res = mc_score_all(pose, config=McConfig(n_samples=8, rng_seed=0), world_up=WORLD_UP)
        det = score_all(pose, world_up=WORLD_UP)
        assert res.baseline.rula.grand_score == det.rula.grand_score
        assert res.baseline.reba.grand_score == det.reba.grand_score

    def test_mc_returns_unit_probabilities(self):
        pose = _high_conf_pose()
        res = mc_score_all(pose, config=McConfig(n_samples=32, rng_seed=0),
                           world_up=WORLD_UP)
        assert abs(res.owas_dist.probs.sum() - 1.0) < 1e-6
        assert abs(res.rula_dist.probs.sum() - 1.0) < 1e-6
        assert abs(res.reba_dist.probs.sum() - 1.0) < 1e-6

    def test_high_confidence_is_peakier(self):
        cfg = McConfig(n_samples=128, sigma_fraction=0.08, rng_seed=0)
        # Use a borderline trunk angle around the 20° RULA threshold
        hi = _high_conf_pose({"trunk_flex_deg": 22.0})
        lo = _low_conf_pose({"trunk_flex_deg": 22.0})
        hi_res = mc_score_all(hi, config=cfg, world_up=WORLD_UP)
        lo_res = mc_score_all(lo, config=cfg, world_up=WORLD_UP)
        # Low-confidence distribution should have larger entropy (more spread)
        assert lo_res.rula_dist.entropy >= hi_res.rula_dist.entropy

    def test_distribution_properties(self):
        pose = _high_conf_pose({"trunk_flex_deg": 22.0})
        res = mc_score_all(pose, config=McConfig(n_samples=64, rng_seed=0),
                           world_up=WORLD_UP)
        for d in (res.owas_dist, res.rula_dist, res.reba_dist):
            assert d.mode in d.classes
            assert d.expected >= d.classes.min() - 1e-6
            assert d.expected <= d.classes.max() + 1e-6
            assert d.variance >= 0
            assert d.entropy >= 0


class TestCalibration:
    def test_temperature_scaling_preserves_argmax_at_T1(self):
        probs = np.array([0.1, 0.6, 0.3])
        out = apply_temperature(probs, T=1.0)
        assert np.argmax(out) == np.argmax(probs)
        assert abs(out.sum() - 1.0) < 1e-9

    def test_temperature_flatter_at_high_T(self):
        probs = np.array([0.1, 0.7, 0.2])
        soft = apply_temperature(probs, T=5.0)
        sharp = apply_temperature(probs, T=0.5)
        # large T → entropy increases
        assert -(soft * np.log(soft + 1e-9)).sum() > -(sharp * np.log(sharp + 1e-9)).sum()

    def test_brier_zero_for_correct_one_hot(self):
        dist = RiskDistribution(
            method="rula",
            classes=np.array([1, 2, 3]),
            probs=np.array([0.0, 1.0, 0.0]),
            n_samples=10,
        )
        b = brier_score([dist], [2])
        assert b == 0.0

    def test_brier_max_for_wrong_certain(self):
        dist = RiskDistribution(
            method="rula", classes=np.array([1, 2, 3]),
            probs=np.array([0.0, 1.0, 0.0]), n_samples=10,
        )
        b = brier_score([dist], [1])   # truth=1, predicted certainly=2
        # (1-0)² + (0-1)² + (0-0)² = 2
        assert b == pytest.approx(2.0)

    def test_ece_perfectly_calibrated_zero(self):
        # 10 samples, all certain at 1.0 and all correct → ECE = 0
        dists = []
        for _ in range(10):
            dists.append(RiskDistribution(
                method="rula", classes=np.array([1, 2]),
                probs=np.array([0.0, 1.0]), n_samples=1,
            ))
        labels = [2] * 10
        assert expected_calibration_error(dists, labels, n_bins=10) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
