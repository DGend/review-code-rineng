"""Tests for `ace_pose.xai` (V — explainable risk attribution)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.ergo_v2 import score_all
from ace_pose.xai import (
    OWAS_AXES, REBA_AXES, RULA_AXES,
    _eval_owas, _eval_rula, _eval_reba,
    attribute_all, attribute_owas, attribute_rula, attribute_reba,
    joint_uncertainty_from_mc, marginal_attribution, shapley_attribution,
)
from ace_pose.tests.test_ergo_v2_golden import _upright_skeleton, WORLD_UP


# ----------------------------------------------------------------------
# Evaluator round-trip consistency with ergo_v2
# ----------------------------------------------------------------------

class TestEvaluatorConsistency:
    """The XAI evaluator functions should reproduce ergo_v2 outputs when
    fed the same axis values."""

    def test_owas_evaluator_matches(self):
        pose = _upright_skeleton(trunk_flex_deg=30.0, knee_flex_deg=20.0)
        result = score_all(pose, world_up=WORLD_UP, load_kg=5.0, bilateral=False)
        owas_values = {
            "back": result.owas.back,
            "arms": result.owas.arms,
            "legs": result.owas.legs,
            "load": result.owas.load,
        }
        assert _eval_owas(owas_values) == result.owas.action_category

    def test_rula_evaluator_matches(self):
        pose = _upright_skeleton(trunk_flex_deg=30.0, upper_arm_flex_deg=40.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        rula_values = {
            "upper_arm": result.rula.upper_arm,
            "lower_arm": result.rula.lower_arm,
            "wrist": result.rula.wrist,
            "wrist_twist": result.rula.wrist_twist,
            "neck": result.rula.neck,
            "trunk": result.rula.trunk,
            "legs": result.rula.legs,
        }
        # XAI evaluator does NOT add muscle/load adjustments, so it should
        # reproduce `grand_score` when muscle_a == muscle_b == load_a == load_b == 0.
        # For the synthetic pose above, those are all zero by default.
        assert _eval_rula(rula_values) == result.rula.grand_score

    def test_reba_evaluator_matches(self):
        pose = _upright_skeleton(trunk_flex_deg=30.0, knee_flex_deg=40.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        reba_values = {
            "trunk": result.reba.trunk,
            "neck": result.reba.neck,
            "legs": result.reba.legs,
            "upper_arm": result.reba.upper_arm,
            "lower_arm": result.reba.lower_arm,
            "wrist": result.reba.wrist,
            "load": result.reba.score_a - result.reba.posture_a,
            "coupling": result.reba.score_b - result.reba.posture_b,
        }
        assert _eval_reba(reba_values) == result.reba.score_c


# ----------------------------------------------------------------------
# Counterfactual / marginal attribution
# ----------------------------------------------------------------------

class TestMarginalAttribution:
    def test_minimum_values_yield_zero_attribution(self):
        """When the pose is already at the minimum, marginal Δ should be 0."""
        values = {a: lo for a, (lo, _) in RULA_AXES.items()}
        marg = marginal_attribution(values, RULA_AXES, _eval_rula)
        assert all(abs(v) < 1e-9 for v in marg.values())

    def test_marginal_nonnegative_for_high_pose(self):
        """For a non-minimal pose, each axis's Δ is ≥ 0 — fixing a sub-score
        can never INCREASE the grand score."""
        pose = _upright_skeleton(trunk_flex_deg=50.0, upper_arm_flex_deg=70.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        attr = attribute_rula(result.rula)
        assert all(v >= -1e-9 for v in attr.marginal.values()), attr.marginal


# ----------------------------------------------------------------------
# Shapley properties
# ----------------------------------------------------------------------

class TestShapley:
    def test_efficiency(self):
        """Σ Shapley_i = v(N) - v(∅): all axis contributions add up to
        the difference between the actual grand and the all-minimum grand."""
        pose = _upright_skeleton(trunk_flex_deg=40.0, upper_arm_flex_deg=60.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        values_actual = {
            a: getattr(result.rula, a) for a in RULA_AXES if a != "wrist_twist"
        }
        values_actual["wrist_twist"] = result.rula.wrist_twist
        mins = {a: lo for a, (lo, _) in RULA_AXES.items()}
        delta = _eval_rula(values_actual) - _eval_rula(mins)
        shap = shapley_attribution(values_actual, RULA_AXES, _eval_rula)
        assert abs(sum(shap.values()) - delta) < 1e-6, (sum(shap.values()), delta)

    def test_dummy_property(self):
        """If an axis value equals its minimum, its Shapley value is 0
        (it never contributes anything to any coalition)."""
        values = {
            "upper_arm": 1,  # min
            "lower_arm": 2,
            "wrist": 3,
            "wrist_twist": 1,  # min
            "neck": 4,
            "trunk": 4,
            "legs": 2,
        }
        shap = shapley_attribution(values, RULA_AXES, _eval_rula)
        assert abs(shap["upper_arm"]) < 1e-9
        assert abs(shap["wrist_twist"]) < 1e-9


# ----------------------------------------------------------------------
# Joint mapping
# ----------------------------------------------------------------------

class TestJointMapping:
    def test_attribution_returns_named_joints(self):
        pose = _upright_skeleton(trunk_flex_deg=40.0, upper_arm_flex_deg=60.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        attr = attribute_all(result)
        for ra in (attr.owas, attr.rula, attr.reba):
            assert ra.joint_shapley, f"{ra.method} missing joint shapley"
            # joints should be from the major-25 layout
            for j in ra.joint_shapley:
                assert "_" in j or j in {"NOSE", "NECK", "PELVIS"}

    def test_joint_marginal_aggregates_axis_contributions(self):
        pose = _upright_skeleton(trunk_flex_deg=50.0, upper_arm_flex_deg=70.0)
        result = score_all(pose, world_up=WORLD_UP, bilateral=False)
        attr = attribute_rula(result.rula)
        # Sum of joint contributions should be ≥ 0 and finite
        s = sum(attr.joint_marginal.values())
        assert s >= 0 and np.isfinite(s)


# ----------------------------------------------------------------------
# Uncertainty decomposition
# ----------------------------------------------------------------------

class TestUncertaintyDecomposition:
    def test_zero_when_no_variance(self):
        from ace_pose.major_joints import MAJOR25_NAMES
        out = joint_uncertainty_from_mc(
            mc_score_samples={"owas": [2, 2, 2], "rula": [3, 3], "reba": [5]},
            pose_keypoint_confidence=np.ones(25),
            major25_names=list(MAJOR25_NAMES),
        )
        # all confidences = 1 → all weights = 0 → all values = 0
        assert all(v == 0.0 for v in out.values())

    def test_low_confidence_gets_more_uncertainty(self):
        from ace_pose.major_joints import MAJOR25_NAMES
        conf = np.ones(25)
        conf[0] = 0.1   # NOSE has low confidence
        out = joint_uncertainty_from_mc(
            mc_score_samples={"rula": [2, 3, 4, 5]},
            pose_keypoint_confidence=conf,
            major25_names=list(MAJOR25_NAMES),
        )
        nose_val = out["NOSE"]
        other_vals = [v for k, v in out.items() if k != "NOSE"]
        assert nose_val > max(other_vals) - 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
