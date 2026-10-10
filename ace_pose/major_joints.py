"""Major-joint subset filter for RTMW3D 133-keypoint output.

Per the project's design decision (2026-05-14): we use whole-body 3D
pose from RTMW3D but **exclude distal finger keypoints**, focusing on
major joints relevant to ergonomic load assessment (wrist included).

This produces a 25-keypoint layout that drives the modernized
OWAS/RULA/REBA scoring code.

Layout (`MAJOR25_NAMES`):
    0  NOSE
    1  EYE_LEFT
    2  EYE_RIGHT
    3  EAR_LEFT
    4  EAR_RIGHT
    5  SHOULDER_LEFT
    6  SHOULDER_RIGHT
    7  ELBOW_LEFT
    8  ELBOW_RIGHT
    9  WRIST_LEFT
    10 WRIST_RIGHT
    11 HIP_LEFT
    12 HIP_RIGHT
    13 KNEE_LEFT
    14 KNEE_RIGHT
    15 ANKLE_LEFT
    16 ANKLE_RIGHT
    17 BIG_TOE_LEFT
    18 SMALL_TOE_LEFT
    19 HEEL_LEFT
    20 BIG_TOE_RIGHT
    21 SMALL_TOE_RIGHT
    22 HEEL_RIGHT
    23 NECK        (derived: midpoint of shoulders)
    24 PELVIS      (derived: midpoint of hips)

We deliberately drop:
- All 68 face landmarks (kept only nose/eyes/ears from body 17pt subset)
- All 21+21 = 42 hand finger keypoints

This preserves wrist (for RULA wrist subscore from forearm geometry),
all body major joints, and feet (needed for OWAS leg posture).

NECK and PELVIS are computed because RTMW3D's COCO-WholeBody does not
expose them as native joints (they are convention-derived in the
RULA/REBA literature).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAJOR25_NAMES = (
    "NOSE",
    "EYE_LEFT", "EYE_RIGHT",
    "EAR_LEFT", "EAR_RIGHT",
    "SHOULDER_LEFT", "SHOULDER_RIGHT",
    "ELBOW_LEFT", "ELBOW_RIGHT",
    "WRIST_LEFT", "WRIST_RIGHT",
    "HIP_LEFT", "HIP_RIGHT",
    "KNEE_LEFT", "KNEE_RIGHT",
    "ANKLE_LEFT", "ANKLE_RIGHT",
    "BIG_TOE_LEFT", "SMALL_TOE_LEFT", "HEEL_LEFT",
    "BIG_TOE_RIGHT", "SMALL_TOE_RIGHT", "HEEL_RIGHT",
    "NECK", "PELVIS",
)

# Index lookups for convenience
NOSE, EYE_L, EYE_R, EAR_L, EAR_R = 0, 1, 2, 3, 4
SHOULDER_L, SHOULDER_R = 5, 6
ELBOW_L, ELBOW_R = 7, 8
WRIST_L, WRIST_R = 9, 10
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14
ANKLE_L, ANKLE_R = 15, 16
BIG_TOE_L, SMALL_TOE_L, HEEL_L = 17, 18, 19
BIG_TOE_R, SMALL_TOE_R, HEEL_R = 20, 21, 22
NECK, PELVIS = 23, 24


# --- COCO-WholeBody-133 index map (for clarity) ---
#  0..16    body (17): COCO-17 layout
#  17..22   feet (6): big_toe_l, small_toe_l, heel_l, big_toe_r, small_toe_r, heel_r
#  23..90   face (68)
#  91..111  left hand (21)  ← excluded
#  112..132 right hand (21) ← excluded
_COCO_BODY_INDICES = list(range(0, 17))        # 17 body
_COCO_FEET_INDICES = list(range(17, 23))       # 6 feet
# We do not copy face or hand keypoints into the major-25 set.


@dataclass
class Major25Pose:
    keypoints: np.ndarray         # (25, 3) — xyz
    confidence: np.ndarray        # (25,)  — per-joint confidence in [0, 1]
    source: np.ndarray            # (25,) int — 0 = direct copy from 133pt, 1 = derived midpoint


def coco_wholebody_to_major25(
    kpts_133: np.ndarray, scores_133: np.ndarray | None = None
) -> Major25Pose:
    """Project the 133-keypoint COCO-WholeBody pose to the 25-keypoint
    major-joint layout. Finger keypoints are discarded.

    Args:
        kpts_133: (133, 3) RTMW3D output. xy may be in pixel scale and z in
            normalised depth — the major-25 layout preserves these as-is;
            downstream code is expected to do scale calibration.
        scores_133: optional (133,) per-keypoint confidence in [0, 1].

    Returns:
        Major25Pose dataclass with (25, 3) keypoints and matched confidence.
    """
    if kpts_133.ndim == 3 and kpts_133.shape[0] == 1:
        kpts_133 = kpts_133[0]
    if scores_133 is not None and scores_133.ndim == 2 and scores_133.shape[0] == 1:
        scores_133 = scores_133[0]

    kp = np.asarray(kpts_133, dtype=np.float64)
    if kp.shape != (133, 3):
        raise ValueError(f"expected (133, 3), got {kp.shape}")

    out = np.zeros((25, 3), dtype=np.float64)
    src = np.zeros(25, dtype=np.int8)

    # 0..16 body
    for j in range(17):
        out[j] = kp[j]
        src[j] = 0
    # 17..22 feet
    for j, coco_idx in enumerate(_COCO_FEET_INDICES, start=17):
        out[j] = kp[coco_idx]
        src[j] = 0

    # 23 NECK — midpoint of shoulders (COCO indices 5, 6)
    out[NECK] = 0.5 * (kp[5] + kp[6])
    src[NECK] = 1
    # 24 PELVIS — midpoint of hips (COCO indices 11, 12)
    out[PELVIS] = 0.5 * (kp[11] + kp[12])
    src[PELVIS] = 1

    if scores_133 is None:
        return Major25Pose(out, np.ones(25), src)

    s = np.asarray(scores_133, dtype=np.float64)
    if s.shape != (133,):
        raise ValueError(f"expected scores (133,), got {s.shape}")
    conf = np.zeros(25, dtype=np.float64)
    for j in range(17):
        conf[j] = float(s[j])
    for j, coco_idx in enumerate(_COCO_FEET_INDICES, start=17):
        conf[j] = float(s[coco_idx])
    # Derived joints: take the geometric mean of parents.
    conf[NECK] = float(np.sqrt(max(s[5], 1e-9) * max(s[6], 1e-9)))
    conf[PELVIS] = float(np.sqrt(max(s[11], 1e-9) * max(s[12], 1e-9)))

    return Major25Pose(out, conf, src)


# Convenience constants for the modernized ergo code: which keypoints are
# required for each ergonomic method (subset of the 25). All RULA/REBA/OWAS
# methods only touch joints inside this 25-point set.
RULA_REQUIRED = (
    SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R,
    NECK, NOSE, HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R,
    PELVIS,
)
REBA_REQUIRED = RULA_REQUIRED
OWAS_REQUIRED = (
    SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R,
    HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R,
    BIG_TOE_L, BIG_TOE_R, NECK, PELVIS,
)
