# PWW-DISCLOSURE: public
"""Which scoring items a single image can measure.

The ergonomics expert scores only joints that are visible and does not analyse an
assessment that has an unmeasured item (expert answers 3-1, 3-2, 4-2). An item is
unmeasurable when one of its joints has pose confidence below JOINT_CONF_MIN, and
the trunk and neck are also unmeasurable away from a side view, because flexion
toward or away from the camera does not show in the image plane.
"""
import numpy as np

from ace_pose.major_joints import (ANKLE_L, ANKLE_R, EAR_L, EAR_R, ELBOW_L, ELBOW_R, EYE_L, EYE_R, HIP_L, HIP_R,
                                   KNEE_L, KNEE_R, SHOULDER_L, SHOULDER_R, WRIST_L, WRIST_R, Major25Pose)

JOINT_CONF_MIN = 0.3
# shoulder width / leg length; frontally about 0.23 H / 0.49 H = 0.47, so 0.24 is
# about 30 degrees from a side view (sin 30 x 0.47). Leg length is the reference
# because the trunk shortens in the image when the worker bends.
SIDE_VIEW_MAX = 0.24
REQUIRED = {"OWAS": ("trunk", "legs", "arm_left", "arm_right"),
            "RULA": ("trunk", "neck", "legs", "arm"),
            "REBA": ("trunk", "neck", "legs", "arm")}


def side_view_ratio(pose: Major25Pose) -> float:
    kp = pose.keypoints[:, :2]
    leg = np.mean([np.linalg.norm(kp[h] - kp[k]) + np.linalg.norm(kp[k] - kp[a])
                   for h, k, a in ((HIP_L, KNEE_L, ANKLE_L), (HIP_R, KNEE_R, ANKLE_R))])
    return float(np.linalg.norm(kp[SHOULDER_L] - kp[SHOULDER_R]) / leg) if leg > 0 else np.inf


def unmeasurable_items(pose: Major25Pose, side_view_max: float = SIDE_VIEW_MAX) -> list[str]:
    seen = pose.confidence >= JOINT_CONF_MIN
    side = side_view_ratio(pose) <= side_view_max
    head = seen[[EAR_L, EAR_R, EYE_L, EYE_R]].any()   # the neck angle uses the ears, else the eyes
    ok = {
        "trunk": side and seen[[SHOULDER_L, SHOULDER_R, HIP_L, HIP_R]].all(),
        "neck": side and seen[[SHOULDER_L, SHOULDER_R]].all() and head,
        "legs": seen[[HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R]].all(),
        "arm_left": seen[[SHOULDER_L, ELBOW_L, WRIST_L]].all(),
        "arm_right": seen[[SHOULDER_R, ELBOW_R, WRIST_R]].all(),
    }
    return [k for k, v in ok.items() if not v]


def measurable(items: list[str], standard: str) -> bool:
    """True when every item the standard needs is measurable (either arm suffices for RULA/REBA)."""
    for need in REQUIRED[standard]:
        if need == "arm":
            if "arm_left" in items and "arm_right" in items:
                return False
        elif need in items:
            return False
    return True
