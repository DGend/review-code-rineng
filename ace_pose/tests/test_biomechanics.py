"""Tests for `ace_pose.biomechanics` (X - inverse dynamics)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.biomechanics import (
    SEGMENTS, TARGET_JOINTS, aggregate_torques, compute_static_moments,
    headline_torques,
)
from ace_pose.tests.test_ergo_v2_golden import _upright_skeleton, WORLD_UP


class TestSegmentParameters:
    def test_mass_fractions_sum_close_to_one(self):
        # head + trunk + 2 arms (upper/forearm/hand) + 2 legs
        total = sum(s.mass_fraction for s in SEGMENTS)
        assert 0.95 < total < 1.05, total

    def test_com_fractions_in_range(self):
        for s in SEGMENTS:
            assert 0.0 <= s.com_fraction <= 1.0, s.name


class TestStaticMoments:
    def test_upright_pose_has_low_torques(self):
        """A perfectly upright worker should have small net joint
        moments because gravity passes near each joint axis."""
        pose = _upright_skeleton()
        r = compute_static_moments(pose, body_mass_kg=70.0,
                                   length_scale_m=1.0, world_up=WORLD_UP)
        assert "L5S1" in r.moments
        # All upright torques should be small (gravity is vertical and
        # body segments stack roughly vertically above the joint).
        # Allow up to 30 N·m baseline (model is approximate).
        for name, m in r.moments.items():
            assert m.torque_Nm < 30.0, f"{name}: {m.torque_Nm:.1f} Nm"

    def test_forward_bend_increases_L5S1(self):
        """Trunk flex 60° should dramatically increase L5/S1 torque."""
        up = _upright_skeleton()
        bent = _upright_skeleton(trunk_flex_deg=60.0)
        ru = compute_static_moments(up, body_mass_kg=70.0, length_scale_m=1.0,
                                    world_up=WORLD_UP)
        rb = compute_static_moments(bent, body_mass_kg=70.0, length_scale_m=1.0,
                                    world_up=WORLD_UP)
        assert rb.moments["L5S1"].torque_Nm > 2 * ru.moments["L5S1"].torque_Nm + 5

    def test_arm_raise_increases_shoulder(self):
        """Raising the arm 90° forward → shoulder torque rises."""
        up = _upright_skeleton()
        raised = _upright_skeleton(upper_arm_flex_deg=90.0)
        ru = compute_static_moments(up, body_mass_kg=70.0, length_scale_m=1.0,
                                    world_up=WORLD_UP)
        rr = compute_static_moments(raised, body_mass_kg=70.0,
                                    length_scale_m=1.0, world_up=WORLD_UP)
        # The 'right' shoulder torque should grow because it now carries
        # the upper arm horizontally.
        assert rr.moments["shoulder_r"].torque_Nm > ru.moments["shoulder_r"].torque_Nm + 1.0

    def test_dimensionless_torque_is_finite(self):
        pose = _upright_skeleton(trunk_flex_deg=40.0)
        r = compute_static_moments(pose, body_mass_kg=70.0, length_scale_m=1.0,
                                   world_up=WORLD_UP)
        for m in r.moments.values():
            assert np.isfinite(m.torque_normalized)
            assert m.torque_normalized >= 0.0

    def test_external_load_increases_torque(self):
        """Adding a 10 kg load at the right hand should increase the
        shoulder torque vs the unloaded pose."""
        pose = _upright_skeleton(upper_arm_flex_deg=90.0)
        kp = pose.keypoints
        wrist_pos = kp[10]   # WRIST_R = index 10
        unloaded = compute_static_moments(pose, body_mass_kg=70.0,
                                          length_scale_m=1.0,
                                          world_up=WORLD_UP)
        loaded = compute_static_moments(
            pose, body_mass_kg=70.0, length_scale_m=1.0, world_up=WORLD_UP,
            external_loads={"shoulder_r": (10.0, wrist_pos)},
        )
        assert (loaded.moments["shoulder_r"].torque_Nm
                > unloaded.moments["shoulder_r"].torque_Nm + 5.0)

    def test_body_mass_scales_torque(self):
        """Doubling body mass should approximately double torques at any
        joint that is currently loaded (non-zero torque)."""
        pose = _upright_skeleton(trunk_flex_deg=40.0)
        r70 = compute_static_moments(pose, body_mass_kg=70.0,
                                     length_scale_m=1.0, world_up=WORLD_UP)
        r140 = compute_static_moments(pose, body_mass_kg=140.0,
                                      length_scale_m=1.0, world_up=WORLD_UP)
        checked = 0
        for name in r70.moments:
            t70 = r70.moments[name].torque_Nm
            if t70 < 1.0:
                continue  # joint with near-zero unloaded torque
            ratio = r140.moments[name].torque_Nm / t70
            assert 1.9 < ratio < 2.1, f"{name}: ratio={ratio}"
            checked += 1
        assert checked > 0, "no joint had a non-trivial torque to scale"


class TestAggregateAndHeadline:
    def test_aggregate_keys(self):
        poses = [_upright_skeleton(trunk_flex_deg=t) for t in (0, 20, 40, 60)]
        results = [
            compute_static_moments(p, body_mass_kg=70.0, length_scale_m=1.0,
                                   world_up=WORLD_UP) for p in poses
        ]
        agg = aggregate_torques(results)
        for j in TARGET_JOINTS:
            assert j in agg
            for stat in ("mean", "median", "std", "min", "max"):
                assert stat in agg[j]

    def test_headline_compact_dict(self):
        pose = _upright_skeleton(trunk_flex_deg=40.0)
        r = compute_static_moments(pose, body_mass_kg=70.0, length_scale_m=1.0,
                                   world_up=WORLD_UP)
        h = headline_torques(r)
        assert "biomech_L5S1_Nm" in h
        assert "biomech_shoulder_max_Nm" in h
        assert "biomech_knee_max_Nm" in h
        assert "biomech_total_load_kg" in h
        # head + trunk + arms + legs ≈ body mass
        assert 0.9 * 70.0 < h["biomech_total_load_kg"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
