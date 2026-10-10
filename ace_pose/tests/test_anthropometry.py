"""Tests for `ace_pose.anthropometry` (C2 scale-invariant correction)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.anthropometry import (
    MEDIAN_STATURE_CM, compute_anthropometry, limb_length_consistency_ok,
    stature_scale_factor,
)
from ace_pose.tests.test_ergo_v2_golden import _upright_skeleton


class TestAnthropometry:
    def test_neutral_pose_reasonable_measurements(self):
        pose = _upright_skeleton()
        a = compute_anthropometry(pose)
        # all positive, in metres
        assert a.shoulder_width > 0.2 and a.shoulder_width < 0.6
        assert a.torso_length > 0.4 and a.torso_length < 0.7
        assert a.upper_arm_l > 0.2 and a.upper_arm_r > 0.2
        assert a.ref_length == max(a.shoulder_width, a.hip_width, a.torso_length)
        # left/right roughly symmetric for the upright synthetic skeleton
        assert abs(a.upper_arm_l - a.upper_arm_r) < 1e-6
        assert abs(a.thigh_l - a.thigh_r) < 1e-6

    def test_consistency_check_passes_for_normal_pose(self):
        pose = _upright_skeleton()
        a = compute_anthropometry(pose)
        assert limb_length_consistency_ok(a)

    def test_stature_scale_factor(self):
        assert stature_scale_factor(MEDIAN_STATURE_CM) == pytest.approx(1.0)
        assert stature_scale_factor(150.0) == pytest.approx(MEDIAN_STATURE_CM / 150.0)
        assert stature_scale_factor(185.0) == pytest.approx(MEDIAN_STATURE_CM / 185.0)
        # missing data → 1.0
        assert stature_scale_factor(None) == 1.0
        assert stature_scale_factor(0) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
