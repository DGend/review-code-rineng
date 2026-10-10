"""Regressions for the image-plane scorer (2026-10-08).

The RTMW3D keypoints have x, y in image pixels and a root-relative depth z in
metres; mixing them made trunk and neck flexion unmeasurable. The scorer now
measures angles in the image plane, taken as the sagittal plane.
"""
import numpy as np

from ace_pose.biomechanics import _estimate_stature
from ace_pose.confidence_mc import McConfig, mc_score_all
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.major_joints import Major25Pose
from ace_pose.tests.test_ergo_v2_golden import WORLD_UP, _upright_skeleton


def _pixels(pose, px_per_m=200.0, depth_scale=0.002):
    """Pose as the model delivers it: x, y in pixels, depth in other units."""
    k = pose.keypoints.copy()
    k[:, :2] *= px_per_m
    k[:, 2] *= depth_scale
    return Major25Pose(keypoints=k, confidence=pose.confidence, source=pose.source)


def test_flexion_measured_despite_depth_in_other_units():
    r = score_all(_pixels(_upright_skeleton(trunk_flex_deg=40.0)), world_up=WORLD_UP)
    assert abs(r.owas.trunk_flex_deg - 40.0) < 1.0
    assert (r.owas.back, r.rula.trunk, r.reba.trunk) == (2, 3, 3)


def test_trunk_sign_forward_and_backward():
    fwd = score_all(_pixels(_upright_skeleton(trunk_flex_deg=30.0)), world_up=WORLD_UP)
    back = score_all(_pixels(_upright_skeleton(trunk_flex_deg=-30.0)), world_up=WORLD_UP)
    assert fwd.owas.trunk_flex_deg > 25 and back.owas.trunk_flex_deg < -25


def test_neck_measured_relative_to_trunk():
    r = score_all(_pixels(_upright_skeleton(trunk_flex_deg=40.0)), world_up=WORLD_UP)
    assert r.rula.neck == 1 and r.reba.neck == 1          # head in line with the trunk
    r = score_all(_pixels(_upright_skeleton(neck_flex_deg=30.0)), world_up=WORLD_UP)
    assert r.rula.neck == 3


def test_reba_knee_adjustment_skipped_when_sitting():
    # thighs horizontal, shanks vertical: hip and knee flexed about 90 degrees
    k = _upright_skeleton().keypoints.copy()
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        k[knee] = k[hip] + np.array([0.45, 0.0, 0.0])
        k[ankle] = k[knee] + np.array([0.0, 0.45, 0.0])
    seated = Major25Pose(keypoints=k, confidence=np.ones(25), source=np.zeros(25, np.int8))
    r = score_all(_pixels(seated), world_up=WORLD_UP)
    assert r.reba.legs == 1, r.reba.notes


def test_mc_scores_both_sides_like_the_deterministic_scorer():
    pose = _pixels(_upright_skeleton(upper_arm_flex_deg=100.0))
    k = pose.keypoints.copy()
    k[8] = k[6] + [0.0, 60.0, 0.0]          # right arm hangs down, left stays raised
    k[10] = k[8] + [0.0, 56.0, 0.0]
    pose = Major25Pose(keypoints=k, confidence=np.ones(25), source=np.zeros(25, np.int8))
    det = score_all(pose, world_up=WORLD_UP)
    mc = mc_score_all(pose, world_up=WORLD_UP, config=McConfig(n_samples=8))
    assert mc.rula_dist.mode == det.rula.grand_score and mc.reba_dist.mode == det.reba.grand_score


def test_stature_estimate_does_not_shrink_when_bending():
    up = _estimate_stature(_pixels(_upright_skeleton()).keypoints)
    bent = _estimate_stature(_pixels(_upright_skeleton(trunk_flex_deg=80.0)).keypoints)
    assert abs(bent - up) / up < 0.05


def _with_arm(pose, direction):
    """Both upper arms pointing along `direction` (image x, y); forearms continue straight."""
    k = pose.keypoints.copy()
    d = np.array([*direction, 0.0]) / np.linalg.norm(direction)
    for sh, el, wr in ((5, 7, 9), (6, 8, 10)):
        k[el] = k[sh] + 60.0 * d
        k[wr] = k[el] + 56.0 * d
    return Major25Pose(keypoints=k, confidence=pose.confidence, source=pose.source)


def test_hanging_arms_of_a_deeply_bent_worker_are_flexed_relative_to_the_trunk():
    # trunk bent past 90 degrees, arms hanging straight down to the floor (M5 in the pilot: 91)
    bent = _pixels(_upright_skeleton(trunk_flex_deg=95.0))
    r = score_all(_with_arm(bent, (0.0, 1.0)), world_up=WORLD_UP)
    assert 75.0 < r.rula.upper_arm_deg < 115.0 and r.rula.upper_arm >= 3


def test_arm_raised_overhead_slightly_behind_vertical_is_not_extension():
    pose = _pixels(_upright_skeleton())
    ant = np.sign(_pixels(_upright_skeleton(trunk_flex_deg=30.0)).keypoints[23, 0] - pose.keypoints[23, 0])
    r = score_all(_with_arm(pose, (-0.17 * ant, -1.0)), world_up=WORLD_UP)
    assert r.rula.upper_arm == 4 and r.reba.upper_arm == 4


def test_deep_forward_bend_on_straight_legs_is_not_sitting():
    # M5 in the pilot: trunk bent about 90 degrees, one knee bent 70, the other nearly straight
    pose = _pixels(_upright_skeleton(trunk_flex_deg=85.0))
    k = pose.keypoints.copy()
    k[16] = k[14] + 90.0 * np.array([np.sin(np.radians(70)), np.cos(np.radians(70)), 0.0])
    r = score_all(Major25Pose(keypoints=k, confidence=pose.confidence, source=pose.source), world_up=WORLD_UP)
    assert r.owas.legs != 1, r.owas.notes


def test_one_bent_knee_with_both_feet_supporting_is_not_owas_code_3():
    # OWAS 3 = standing on one straight leg; with weight on both legs a single bent knee is code 2
    pose = _pixels(_upright_skeleton())
    k = pose.keypoints.copy()
    k[16] = k[14] + np.array([60.0, 80.0, 0.0])        # right shank angled forward: knee about 37 degrees
    r = score_all(Major25Pose(keypoints=k, confidence=pose.confidence, source=pose.source), world_up=WORLD_UP)
    assert r.owas.legs in (2, 4) and r.owas.legs != 3, r.owas.notes
