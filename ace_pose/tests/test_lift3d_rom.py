"""3D pose from 2D keypoints under bone-length ratios and joint range-of-motion limits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ace_pose import lift3d_rom as L  # noqa: E402
from ace_pose.major_joints import (  # noqa: E402
    ANKLE_R, ELBOW_R, HIP_R, KNEE_R, NECK, PELVIS, SHOULDER_R, WRIST_R,
)


def _params(**kw):
    p = L.neutral_params()
    for k, v in kw.items():
        p[L.PARAM_INDEX[k]] = v
    return p


def _trunk_flexion(j):
    """Lean of pelvis->neck from vertical in the sagittal plane (body y up, z forward), degrees."""
    v = j[NECK] - j[PELVIS]
    return np.degrees(np.arctan2(v[2], v[1]))


def test_neutral_pose_has_standard_proportions():
    j = L.forward(L.neutral_params())
    assert np.isclose(np.linalg.norm(j[SHOULDER_R] - j[ELBOW_R]), L.SEGMENTS["upper_arm"])
    assert np.isclose(np.linalg.norm(j[ELBOW_R] - j[WRIST_R]), L.SEGMENTS["forearm"])
    assert np.isclose(np.linalg.norm(j[HIP_R] - j[KNEE_R]), L.SEGMENTS["thigh"])
    assert np.isclose(np.linalg.norm(j[KNEE_R] - j[ANKLE_R]), L.SEGMENTS["shank"])
    assert abs(_trunk_flexion(j)) < 1e-6


def test_hinges_cannot_bend_backwards():
    lo, hi = L.bounds()
    for name in ("knee_r", "knee_l", "elbow_r", "elbow_l"):
        assert lo[L.PARAM_INDEX[name]] >= -10.0         # at most a few degrees of hyperextension


def test_side_view_trunk_flexion_is_recovered():
    true = _params(root_yaw=90.0, trunk_flex=40.0, shoulder_flex_r=60.0, shoulder_flex_l=60.0)
    kp2d = L.project(L.forward(true), L.camera_rotation(true))
    fit = L.fit(kp2d, np.ones(25))
    flex = [_trunk_flexion(j) for j in fit.body_joints]
    assert abs(np.median(flex) - 40.0) < 6.0


def _front_lean_spread(conf):
    true = _params(trunk_flex=35.0)                     # leaning toward the camera
    fit = L.fit(L.project(L.forward(true), L.camera_rotation(true)), conf)
    f = [_trunk_flexion(j) for j in fit.body_joints]
    return max(f) - min(f)


def test_front_view_without_face_and_feet_keeps_both_lean_directions():
    # toward or away from the camera cannot be told from the trunk and limbs alone
    conf = np.ones(25)
    conf[[0, 1, 2, 3, 4, 17, 18, 19, 20, 21, 22]] = 0.0
    assert _front_lean_spread(conf) > 50.0


def test_face_and_feet_resolve_the_lean_direction():
    assert _front_lean_spread(np.ones(25)) < 5.0


# --- hand segment: forearm rotation, wrist flexion and deviation (VA 38 CFR 4.71 Plate I) ---

def _wrist(p, side):
    j, h = L.forward(p), L.forward_hands(p)
    k = 0 if side == "r" else 1
    el, wr = (j[ELBOW_R], j[WRIST_R]) if side == "r" else (j[7], j[9])
    return L.wrist_angles(el, wr, h[k, 0], h[k, 1], h[k, 2], side)


def test_neutral_wrist_is_zero():
    for side in ("r", "l"):
        assert np.allclose(_wrist(L.neutral_params(), side), (0.0, 0.0), atol=1e-6)


def test_wrist_parameters_read_back_as_angles():
    for side in ("r", "l"):
        p = _params(**{f"wrist_flex_{side}": 30.0, f"wrist_dev_{side}": 15.0, f"forearm_rot_{side}": 40.0,
                       f"elbow_{side}": 60.0, f"shoulder_flex_{side}": 30.0})
        assert np.allclose(_wrist(p, side), (30.0, 15.0), atol=1e-6)


def test_wrist_bounds_follow_va_plate_one():
    lo, hi = L.bounds()
    for s in ("r", "l"):
        assert (lo[L.PARAM_INDEX[f"wrist_flex_{s}"]], hi[L.PARAM_INDEX[f"wrist_flex_{s}"]]) == (-70, 80)
        assert (lo[L.PARAM_INDEX[f"wrist_dev_{s}"]], hi[L.PARAM_INDEX[f"wrist_dev_{s}"]]) == (-20, 45)
        assert (lo[L.PARAM_INDEX[f"forearm_rot_{s}"]], hi[L.PARAM_INDEX[f"forearm_rot_{s}"]]) == (-85, 80)


def test_front_view_wrist_flexion_is_recovered_from_hand_points():
    true = _params(wrist_flex_r=40.0)
    R = L.camera_rotation(true)
    kp2d, hands2d = L.project(L.forward(true), R), L.project(L.forward_hands(true).reshape(6, 3), R)
    fit = L.fit(kp2d, np.ones(25), hands2d=hands2d, hands_conf=np.ones(6))
    flex = [L.wrist_angles(j[ELBOW_R], j[WRIST_R], *h[0], "r")[0] for j, h in zip(fit.body_joints, fit.hand_joints)]
    assert abs(np.median(flex) - 40.0) < 8.0


def test_ankle_dorsiflexion_lifts_the_toes_and_neutral_toes_are_on_the_floor():
    from ace_pose.major_joints import BIG_TOE_R, HEEL_R
    j0, j_up = L.forward(L.neutral_params()), L.forward(_params(ankle_r=20.0))
    assert j_up[BIG_TOE_R, 1] > j0[BIG_TOE_R, 1]               # +ankle = dorsiflexion (VA Plate II)
    assert np.isclose(j0[BIG_TOE_R, 1], j0[HEEL_R, 1])          # standing: toe and heel at sole level
    assert j0[BIG_TOE_R, 1] < j0[ANKLE_R, 1]
