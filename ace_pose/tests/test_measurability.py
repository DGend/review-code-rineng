"""Items whose joints are not seen, or not seen from the side, are not scored (2026-10-08).

The ergonomics expert scores only what is visible and does not analyse an
assessment with an unmeasured item (expert answers 3-1, 3-2, 4-2).
"""
import numpy as np

from ace_pose.ergo_v2.measurability import measurable, unmeasurable_items
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.major_joints import Major25Pose
from ace_pose.tests.test_ergo_v2_golden import WORLD_UP, _upright_skeleton
from ace_pose.tests.test_planar_scoring import _pixels


def _with(pose, keypoints=None, confidence=None):
    return Major25Pose(keypoints=pose.keypoints if keypoints is None else keypoints,
                       confidence=pose.confidence if confidence is None else confidence, source=pose.source)


def test_side_view_with_every_joint_seen_is_measurable():
    pose = _pixels(_upright_skeleton(trunk_flex_deg=30.0))
    assert unmeasurable_items(pose) == []
    assert all(measurable(pose_items, s) for pose_items in [[]] for s in ("OWAS", "RULA", "REBA"))


def test_unseen_knee_makes_the_legs_and_every_method_unmeasurable():
    pose = _pixels(_upright_skeleton())
    c = pose.confidence.copy()
    c[14] = 0.1                                       # right knee hidden
    items = unmeasurable_items(_with(pose, confidence=c))
    assert "legs" in items and not any(measurable(items, s) for s in ("OWAS", "RULA", "REBA"))


def test_front_or_back_view_makes_trunk_and_neck_unmeasurable():
    pose = _pixels(_upright_skeleton())
    k = pose.keypoints.copy()
    mid = (k[5] + k[6]) / 2
    leg = np.linalg.norm(k[11] - k[13]) + np.linalg.norm(k[13] - k[15])
    k[5], k[6] = mid - [0.25 * leg, 0, 0], mid + [0.25 * leg, 0, 0]   # shoulders spread across the image
    items = unmeasurable_items(_with(pose, keypoints=k))
    assert {"trunk", "neck"} <= set(items)


def test_one_hidden_arm_is_scored_from_the_visible_side():
    pose = _pixels(_upright_skeleton(upper_arm_flex_deg=100.0))
    k, c = pose.keypoints.copy(), pose.confidence.copy()
    k[8] = k[6] + [0.0, 60.0, 0.0]                   # right arm hangs down
    k[10] = k[8] + [0.0, 56.0, 0.0]
    c[[7, 9]] = 0.1                                   # left (raised) arm hidden
    r = score_all(_with(pose, keypoints=k, confidence=c), world_up=WORLD_UP)
    assert r.rula.side == "right" and r.rula.upper_arm == 1
    items = r.extras["unmeasurable"]
    assert "arm_left" in items and measurable(items, "RULA") and not measurable(items, "OWAS")
