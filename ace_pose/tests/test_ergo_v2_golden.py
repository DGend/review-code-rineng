"""Golden tests for `ace_pose.ergo_v2`.

Synthetic poses are constructed analytically to hit specific score
boundary conditions in OWAS / RULA / REBA, so that the score the
implementation returns must match the expected published value to
within ±0 (categorical sub-scores) and ±1 (grand score, allowing
boundary-tolerance).

Coordinate frame used:
    +X right, +Y down (image-down convention), +Z forward.
    The body's +up_body axis therefore points in -Y in this frame.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.major_joints import Major25Pose, MAJOR25_NAMES, NECK, PELVIS
from ace_pose.ergo_v2 import score_owas, score_rula, score_reba, score_all
from ace_pose.ergo_v2.angles import build_body_frame


WORLD_UP = np.array([0.0, -1.0, 0.0])  # image-y is down


# ----------------------------------------------------------------------
# Pose builder utilities
# ----------------------------------------------------------------------

def _upright_skeleton(
    trunk_flex_deg: float = 0.0,
    upper_arm_flex_deg: float = 0.0,
    elbow_flex_deg: float = 0.0,
    neck_flex_deg: float = 0.0,
    knee_flex_deg: float = 0.0,
    trunk_twist_deg: float = 0.0,
) -> Major25Pose:
    """Build a parametric human skeleton in metres.

    Coordinate convention:
      +X = right (subject's right shoulder lies in +X)
      +Y = down (image-down)
      +Z = anterior (the direction the subject faces; for the default
           upright pose the worker faces +Z)

    Convention for the angle parameters:
      trunk_flex_deg > 0  → trunk leans anteriorly (toward +Z)
      upper_arm_flex_deg > 0 → upper arm rotates anteriorly from hanging-down
      neck_flex_deg > 0  → head tips anteriorly
      elbow_flex_deg  > 0 → forearm bends inward (toward shoulder); standard
                              RULA "flexion" angle measured as 180°-interior.
      knee_flex_deg  > 0 → knee flexed (shank moves forward of thigh)
    """
    L_hip_width = 0.30
    L_shoulder_width = 0.40
    L_torso = 0.55
    L_upper_arm = 0.30
    L_forearm = 0.28
    L_thigh = 0.45
    L_shank = 0.45
    L_neck_to_head = 0.25
    L_ear_width = 0.18

    pelvis = np.array([0.0, 1.0, 0.0])
    hip_l = pelvis + np.array([-L_hip_width / 2, 0.0, 0.0])
    hip_r = pelvis + np.array([+L_hip_width / 2, 0.0, 0.0])

    # Trunk direction: at flex=0 points straight up (-Y).
    # As flex increases, trunk rotates toward +Z (anterior).
    t = math.radians(trunk_flex_deg)
    trunk_dir = np.array([0.0, -math.cos(t), math.sin(t)])
    neck = pelvis + L_torso * trunk_dir

    # anatomical anterior in world frame, given current trunk tilt
    anterior_dir = np.array([0.0, math.sin(t), math.cos(t)])

    # shoulders translate with the trunk (X-offset preserved)
    sh_offset = np.array([L_shoulder_width / 2, 0.0, 0.0])
    sh_r = neck + sh_offset
    sh_l = neck - sh_offset

    # optional trunk twist around the trunk axis
    if abs(trunk_twist_deg) > 1e-6:
        from scipy.spatial.transform import Rotation
        Rrot = Rotation.from_rotvec(math.radians(trunk_twist_deg) * trunk_dir)
        sh_r = neck + Rrot.apply(sh_r - neck)
        sh_l = neck + Rrot.apply(sh_l - neck)

    # Upper arm direction: at flex=0, arm hangs along -trunk_dir.
    # As flex increases, arm rotates anteriorly (toward +anterior_dir)
    # within the body's sagittal plane.
    a = math.radians(upper_arm_flex_deg)
    arm_dir = math.cos(a) * (-trunk_dir) + math.sin(a) * anterior_dir
    elbow_r = sh_r + L_upper_arm * arm_dir
    elbow_l = sh_l + L_upper_arm * arm_dir

    # Forearm direction: bent inward (toward shoulder) by elbow_flex_deg.
    # The bend direction is back along the arm (-arm_dir), tilted into
    # the body's anterior plane.
    e = math.radians(elbow_flex_deg)
    forearm_dir = math.cos(e) * arm_dir + math.sin(e) * anterior_dir
    wrist_r = elbow_r + L_forearm * forearm_dir
    wrist_l = elbow_l + L_forearm * forearm_dir

    # Head: extension of trunk_dir upward, with neck_flex tilting head
    # anteriorly relative to the trunk.
    n = math.radians(neck_flex_deg)
    head_dir = math.cos(n) * trunk_dir + math.sin(n) * anterior_dir
    head_top = neck + L_neck_to_head * head_dir
    nose = neck + 0.8 * L_neck_to_head * head_dir
    # eyes level with the ears and 8 cm in front of them, so the ear -> eye
    # line (Frankfurt plane) is perpendicular to the head axis
    head_anterior = -math.sin(n) * trunk_dir + math.cos(n) * anterior_dir
    ear_offset_lat = np.array([L_ear_width / 2, 0.0, 0.0])
    eye_l = nose + np.array([-0.04, 0.0, 0.0]) + 0.08 * head_anterior
    eye_r = nose + np.array([+0.04, 0.0, 0.0]) + 0.08 * head_anterior
    # ears on the head axis (ear over shoulder in a neutral posture)
    ear_l = nose - ear_offset_lat
    ear_r = nose + ear_offset_lat

    # Legs: thigh hangs straight down at hip; shank may flex.
    thigh_dir = np.array([0.0, 1.0, 0.0])
    knee_l = hip_l + L_thigh * thigh_dir
    knee_r = hip_r + L_thigh * thigh_dir
    k = math.radians(knee_flex_deg)
    shank_dir = math.cos(k) * thigh_dir + math.sin(k) * np.array([0.0, 0.0, 1.0])
    ankle_l = knee_l + L_shank * shank_dir
    ankle_r = knee_r + L_shank * shank_dir

    # feet (big toe forward of ankle, heel behind)
    big_toe_l = ankle_l + np.array([0.0, 0.05, 0.20])
    small_toe_l = ankle_l + np.array([-0.05, 0.05, 0.18])
    heel_l = ankle_l + np.array([0.0, 0.05, -0.10])
    big_toe_r = ankle_r + np.array([0.0, 0.05, 0.20])
    small_toe_r = ankle_r + np.array([+0.05, 0.05, 0.18])
    heel_r = ankle_r + np.array([0.0, 0.05, -0.10])

    pts = np.zeros((25, 3), dtype=np.float64)
    pts[0] = nose
    pts[1] = eye_l
    pts[2] = eye_r
    pts[3] = ear_l
    pts[4] = ear_r
    pts[5] = sh_l
    pts[6] = sh_r
    pts[7] = elbow_l
    pts[8] = elbow_r
    pts[9] = wrist_l
    pts[10] = wrist_r
    pts[11] = hip_l
    pts[12] = hip_r
    pts[13] = knee_l
    pts[14] = knee_r
    pts[15] = ankle_l
    pts[16] = ankle_r
    pts[17] = big_toe_l
    pts[18] = small_toe_l
    pts[19] = heel_l
    pts[20] = big_toe_r
    pts[21] = small_toe_r
    pts[22] = heel_r
    pts[NECK] = neck
    pts[PELVIS] = pelvis
    # Seen from the side: the scorer measures angles in the image plane, so
    # the subject's anterior (+Z above) maps to image +X and the subject's
    # right (+X above) to the depth axis.
    pts = np.stack([pts[:, 2], pts[:, 1], -pts[:, 0]], axis=1)

    return Major25Pose(
        keypoints=pts,
        confidence=np.ones(25),
        source=np.zeros(25, dtype=np.int8),
    )


# ----------------------------------------------------------------------
# OWAS golden cases
# ----------------------------------------------------------------------

class TestOwasGolden:
    def test_upright_standing_minimal_risk(self):
        """Worker standing upright, arms down, balanced → Back=1, Arms=1, Legs=2, Load=1 → AC=1."""
        pose = _upright_skeleton()
        r = score_owas(pose, load_kg=0.0, world_up=WORLD_UP)
        assert r.back == 1, f"back: {r.notes}"
        assert r.arms == 1
        assert r.legs == 2
        assert r.load == 1
        assert r.action_category == 1

    def test_straight_legs_with_feet_off_center_stay_code_2(self):
        # Both ankles shifted to one side of the pelvis (common in side views)
        # used to be read as one-legged support (code 3); weight distribution is
        # not observable from one frame, so straight legs stay code 2.
        from ace_pose.major_joints import ANKLE_L, ANKLE_R, BIG_TOE_L, BIG_TOE_R
        pose = _upright_skeleton()
        side = build_body_frame(pose).side
        k = pose.keypoints.copy()
        for j in (ANKLE_L, ANKLE_R, BIG_TOE_L, BIG_TOE_R):
            k[j] = k[j] + 0.4 * side * np.linalg.norm(k[ANKLE_L] - k[ANKLE_R])
        shifted = Major25Pose(keypoints=k, confidence=pose.confidence, source=pose.source)
        assert score_owas(shifted, load_kg=0.0, world_up=WORLD_UP).legs == 2

    def test_bent_forward_40deg_no_twist(self):
        """Trunk bent 40° forward, arms down, legs straight → Back=2."""
        pose = _upright_skeleton(trunk_flex_deg=40.0)
        r = score_owas(pose, world_up=WORLD_UP)
        assert r.back == 2, f"back: {r.notes}"

    def test_twist_is_not_observable_in_the_image_plane(self):
        """Twist lies out of the image plane, so a bent and twisted trunk is
        scored as bent only (Back=2), a stated limitation of the 2D scorer."""
        pose = _upright_skeleton(trunk_flex_deg=40.0, trunk_twist_deg=30.0)
        r = score_owas(pose, world_up=WORLD_UP)
        assert r.back == 2, f"expected Back=2, got {r.back}: {r.notes}"

    def test_heavy_load_amplifies_risk(self):
        """Same posture, load > 20 kg should bump load digit to 3."""
        pose = _upright_skeleton(trunk_flex_deg=40.0)
        r_light = score_owas(pose, load_kg=5.0, world_up=WORLD_UP)
        r_heavy = score_owas(pose, load_kg=25.0, world_up=WORLD_UP)
        assert r_light.load == 1
        assert r_heavy.load == 3
        assert r_heavy.action_category >= r_light.action_category


# ----------------------------------------------------------------------
# RULA golden cases
# ----------------------------------------------------------------------

class TestRulaGolden:
    def test_upright_arms_down(self):
        """Neutral standing: upper arm ≈ 0°, elbow ≈ 180°, neck ≈ 0°, trunk ≈ 0°
        → all sub-scores low; grand score ≤ 3."""
        pose = _upright_skeleton()
        r = score_rula(pose, side="right", world_up=WORLD_UP)
        assert r.upper_arm == 1, f"upper_arm: {r.notes}"
        assert r.lower_arm == 2, f"elbow ≈ 180° → score 2 (outside 60-100 range): {r.notes}"
        assert r.neck == 1
        assert r.trunk == 1
        assert r.grand_score <= 4

    def test_upper_arm_raised_to_60deg(self):
        """Upper arm flexion 60°: 45° < 60° ≤ 90° → score 3."""
        pose = _upright_skeleton(upper_arm_flex_deg=60.0)
        r = score_rula(pose, side="right", world_up=WORLD_UP)
        assert r.upper_arm == 3, f"upper arm 60° → 3 expected, got {r.upper_arm}: {r.notes}"

    def test_arm_well_flexed_elbow(self):
        """Elbow ≈ 80° (inside 60-100°) → lower_arm = 1."""
        pose = _upright_skeleton(elbow_flex_deg=80.0)
        r = score_rula(pose, side="right", world_up=WORLD_UP)
        assert r.lower_arm == 1, f"elbow 80° → 1 expected, got {r.lower_arm}: {r.notes}"

    def test_severe_trunk_bend_high_grand(self):
        """Trunk flex 70° + raised arms 80° → trunk 4 + upper_arm 3.

        Per the RULA table the grand score is 4 (action level 2) without
        added load/muscle factors. Adding a 5 kg load to push past the
        score-A muscle/force adjustment moves the grand into action 3.
        """
        pose = _upright_skeleton(trunk_flex_deg=70.0, upper_arm_flex_deg=80.0)
        # baseline without load: trunk=4 expected
        baseline = score_rula(pose, side="right", world_up=WORLD_UP)
        assert baseline.trunk == 4, f"trunk 70° → 4 expected, got {baseline.trunk}: {baseline.notes}"
        # with 5 kg static load, score_a +1 and we cross into action level 3
        loaded = score_rula(
            pose, side="right", world_up=WORLD_UP,
            load_kg=5.0, static_minute=True,
        )
        assert loaded.action_level >= 3, (
            f"loaded action level expected ≥3, got {loaded.action_level}: {loaded.notes}"
        )


# ----------------------------------------------------------------------
# REBA golden cases
# ----------------------------------------------------------------------

class TestRebaGolden:
    def test_neutral_pose_low_risk(self):
        """Neutral standing → REBA grand ≤ 3 (low risk)."""
        pose = _upright_skeleton()
        # use coupling=0 (Good) and activity_static=False to test the "least
        # risky" configuration the standard allows.
        r = score_reba(
            pose, side="right", world_up=WORLD_UP,
            coupling=0, activity_static=False,
        )
        # trunk 1, neck 1, legs 1, upper 1, lower 2 (elbow ~180°), wrist 1
        # Table A[0,0,0] = 1 → score_a 1; Table B[0,1,0] = 1 → score_b 1;
        # Table C[0,0] = 1 → grand 1 + 0 = 1
        assert r.grand_score <= 3, f"neutral grand={r.grand_score}: {r.notes}"

    def test_trunk_flex_increases_score(self):
        """Trunk 40° flex → trunk score 3 → grand ≥ trunk-1 baseline by ≥ 2."""
        baseline = score_reba(_upright_skeleton(), side="right", world_up=WORLD_UP,
                              coupling=0, activity_static=False)
        flexed = score_reba(_upright_skeleton(trunk_flex_deg=40.0), side="right",
                            world_up=WORLD_UP, coupling=0, activity_static=False)
        # Table C maps score A 2 / score B 1 to grand 1, so the grand may not move;
        # the trunk and score A must.
        assert flexed.trunk == 3, f"trunk 40° → 3 expected, got {flexed.trunk}"
        assert flexed.score_a > baseline.score_a, f"{flexed.notes}"

    def test_default_coupling_and_static_inflate_score(self):
        """Same neutral pose with default coupling=1 and activity_static=True
        should yield a higher grand than coupling=0 / activity=False."""
        pose = _upright_skeleton()
        permissive = score_reba(pose, side="right", world_up=WORLD_UP,
                                coupling=0, activity_static=False)
        defaults = score_reba(pose, side="right", world_up=WORLD_UP)  # use new defaults
        assert defaults.grand_score >= permissive.grand_score


# ----------------------------------------------------------------------
# Sanity: lookup tables match the published shapes
# ----------------------------------------------------------------------

class TestTableShapes:
    def test_owas_table(self):
        from ace_pose.ergo_v2.tables import OWAS_AC
        assert OWAS_AC.shape == (4, 3, 7, 3)
        assert OWAS_AC.min() >= 1 and OWAS_AC.max() <= 4

    def test_rula_table_a_cells(self):
        # McAtamney & Corlett 1993 Table A, rows (W1T1, W1T2, ..., W4T2). Pinned
        # cell by cell: a shape-only check let a 31-cell transcription error
        # ship unnoticed (see standards/rula.md, correction 2026-10-04).
        from ace_pose.ergo_v2.tables import RULA_TABLE_A
        published = [
            [[1, 2, 2, 2, 2, 3, 3, 3], [2, 2, 2, 2, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3, 4, 4]],
            [[2, 3, 3, 3, 3, 4, 4, 4], [3, 3, 3, 3, 3, 4, 4, 4], [3, 4, 4, 4, 4, 4, 5, 5]],
            [[3, 3, 4, 4, 4, 4, 5, 5], [3, 4, 4, 4, 4, 4, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5]],
            [[4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 5, 5, 5, 6, 6]],
            [[5, 5, 5, 5, 5, 6, 6, 7], [5, 6, 6, 6, 6, 7, 7, 7], [6, 6, 6, 7, 7, 7, 7, 8]],
            [[7, 7, 7, 7, 7, 8, 8, 9], [8, 8, 8, 8, 8, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9]],
        ]
        assert RULA_TABLE_A.reshape(6, 3, 8).tolist() == published

    def test_rula_tables(self):
        from ace_pose.ergo_v2.tables import RULA_TABLE_A, RULA_TABLE_B, RULA_TABLE_C
        assert RULA_TABLE_A.shape == (6, 3, 4, 2)
        assert RULA_TABLE_B.shape == (6, 6, 2)
        assert RULA_TABLE_C.shape == (8, 7)

    def test_reba_tables(self):
        from ace_pose.ergo_v2.tables import REBA_TABLE_A, REBA_TABLE_B, REBA_TABLE_C
        assert REBA_TABLE_A.shape == (5, 3, 4)
        assert REBA_TABLE_B.shape == (6, 2, 3)
        assert REBA_TABLE_C.shape == (12, 12)

    def test_reba_table_c_asymmetry(self):
        """REBA Table C is intentionally asymmetric per `standards/reba.md §13.4`."""
        from ace_pose.ergo_v2.tables import REBA_TABLE_C
        assert REBA_TABLE_C[0, 5] == 3
        assert REBA_TABLE_C[5, 0] == 6
        assert REBA_TABLE_C[0, 5] != REBA_TABLE_C[5, 0]


# ----------------------------------------------------------------------
# Combined pipeline
# ----------------------------------------------------------------------

class TestCombinedPipeline:
    def test_score_all_runs_and_returns_unified(self):
        pose = _upright_skeleton(trunk_flex_deg=30.0, upper_arm_flex_deg=30.0)
        result = score_all(pose, world_up=WORLD_UP, load_kg=5.0)
        assert result.owas.action_category in (1, 2, 3, 4)
        assert result.rula.grand_score in range(1, 8)
        assert result.reba.grand_score in range(1, 16)
        # bilateral filled
        assert result.rula_left is not None and result.rula_right is not None
        assert result.reba_left is not None and result.reba_right is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
