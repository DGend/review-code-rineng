"""C2 — Scale-invariant anthropometric correction.

Replaces absolute-distance thresholds in ergo scoring with limb-length
ratios that are robust to:
  - Worker stature variation (Korean cohort spans ~150-185 cm)
  - Camera-to-subject distance variation (RTMW3D returns pixel-scale x/y
    plus normalised depth z, so absolute distances are not directly
    comparable across frames)

Per `standards/owas.md §4` and audit recommendations, only a small
number of ergo decisions use ABSOLUTE distances (OWAS lower-limb
distance, RULA out-to-side check). Angles are scale-invariant and
unaffected by this module.

Strategy
--------
1. From a pose, derive a stable reference length L_ref defined as
   max(shoulder_width, hip_width, torso_length). This is the most
   reliable scale signal across postures.
2. All distance-based thresholds become RATIOS of L_ref. The legacy
   absolute thresholds (60 mm, 100 mm, 30 cm) are converted to the
   ratio value that holds in the median anthropometry, then re-applied
   per pose using each pose's measured L_ref.
3. If the worker's stature (cm) is available, use it as an additional
   covariate: scale the ratio threshold by `stature / median_stature`.
   This is optional and falls back gracefully to pure ratios when
   stature is missing.

This module exposes:
  - `compute_anthropometry(pose)` → AnthropometryReport
  - `is_arm_out_to_side_ratio(...)`, `is_knee_at_floor_ratio(...)` etc.
  - `adjust_thresholds(stature_cm, anthro)` → dict of per-pose thresholds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ace_pose.major_joints import (
    Major25Pose,
    ANKLE_L, ANKLE_R, BIG_TOE_L, BIG_TOE_R,
    ELBOW_L, ELBOW_R, HIP_L, HIP_R, KNEE_L, KNEE_R,
    NECK, PELVIS, SHOULDER_L, SHOULDER_R, WRIST_L, WRIST_R,
)


# Median adult stature for Korean industrial cohort (per project metadata,
# PDF page 4: 7-factory worker heights, mean ≈ 172 cm).
MEDIAN_STATURE_CM = 172.0


@dataclass
class AnthropometryReport:
    shoulder_width: float
    hip_width: float
    torso_length: float
    upper_arm_l: float
    upper_arm_r: float
    forearm_l: float
    forearm_r: float
    thigh_l: float
    thigh_r: float
    shank_l: float
    shank_r: float
    ref_length: float            # max(shoulder, hip, torso) — the scale base
    # ratios useful for downstream:
    arm_total_l: float           # upper_arm_l + forearm_l
    arm_total_r: float
    leg_total_l: float
    leg_total_r: float
    stature_proxy: float         # rough proxy if stature_cm unavailable

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


def compute_anthropometry(pose: Major25Pose) -> AnthropometryReport:
    kp = pose.keypoints
    sh_w = float(np.linalg.norm(kp[SHOULDER_R] - kp[SHOULDER_L]))
    hi_w = float(np.linalg.norm(kp[HIP_R] - kp[HIP_L]))
    torso = float(np.linalg.norm(kp[NECK] - kp[PELVIS]))
    ua_l = float(np.linalg.norm(kp[ELBOW_L] - kp[SHOULDER_L]))
    ua_r = float(np.linalg.norm(kp[ELBOW_R] - kp[SHOULDER_R]))
    fa_l = float(np.linalg.norm(kp[WRIST_L] - kp[ELBOW_L]))
    fa_r = float(np.linalg.norm(kp[WRIST_R] - kp[ELBOW_R]))
    th_l = float(np.linalg.norm(kp[KNEE_L] - kp[HIP_L]))
    th_r = float(np.linalg.norm(kp[KNEE_R] - kp[HIP_R]))
    sh_l = float(np.linalg.norm(kp[ANKLE_L] - kp[KNEE_L]))
    sh_r = float(np.linalg.norm(kp[ANKLE_R] - kp[KNEE_R]))

    ref = max(sh_w, hi_w, torso)
    arm_l = ua_l + fa_l
    arm_r = ua_r + fa_r
    leg_l = th_l + sh_l
    leg_r = th_r + sh_r
    stature_proxy = torso + 0.5 * (leg_l + leg_r) + 0.25 * (arm_l + arm_r) * 0  # legs+torso

    return AnthropometryReport(
        shoulder_width=sh_w, hip_width=hi_w, torso_length=torso,
        upper_arm_l=ua_l, upper_arm_r=ua_r,
        forearm_l=fa_l, forearm_r=fa_r,
        thigh_l=th_l, thigh_r=th_r,
        shank_l=sh_l, shank_r=sh_r,
        ref_length=max(ref, 1e-6),
        arm_total_l=arm_l, arm_total_r=arm_r,
        leg_total_l=leg_l, leg_total_r=leg_r,
        stature_proxy=stature_proxy,
    )


# ----------------------------------------------------------------------
# Threshold conversions
# ----------------------------------------------------------------------

# Conversion ratios derived from a 1.72-m worker:
# - OWAS knee-to-toe (was 60 mm absolute)       → 0.10 × leg_total
# - RULA "arm out to side" margin (was 15 %)     → 0.15 × shoulder_width
# - OWAS lower-limb pose-error sanity (100 mm)  → 0.20 × torso_length
#
# These constants live here so the audit doc can cross-reference them.

OWAS_KNEE_GROUND_RATIO = 0.10           # vs leg_total
OWAS_LOWER_DIFF_SANITY_RATIO = 0.20     # vs torso_length
RULA_ARM_OUT_TO_SIDE_RATIO = 0.15       # vs shoulder_width
RULA_WRIST_REACH_RATIO = 1.0            # vs upper-arm length (signals over-reach)


def stature_scale_factor(stature_cm: float | None,
                         median: float = MEDIAN_STATURE_CM) -> float:
    """Multiplier that scales pure ratios by individual stature.

    Workers shorter than median have a slightly LARGER permissible
    distance-ratio (their absolute reach is smaller, so the *ratio* of
    a fixed-environment workstation distance to their reach is larger).
    Returns 1.0 when stature is unknown.
    """
    if stature_cm is None or stature_cm <= 0:
        return 1.0
    return median / float(stature_cm)


def is_arm_out_to_side(
    pose: Major25Pose, anthro: AnthropometryReport, side: str = "right",
    stature_cm: float | None = None,
) -> bool:
    """Per `standards/rula.md §3.2`: arm out to side adds +1 to lower-arm.

    The legacy code uses a 15 % overshoot of wrist lateral vs shoulder
    lateral. We make that 15 % into a ratio of `shoulder_width` and
    optionally scale by stature.
    """
    from ace_pose.ergo_v2.angles import build_body_frame
    frame = build_body_frame(pose)
    kp = pose.keypoints
    sh = kp[SHOULDER_R] if side == "right" else kp[SHOULDER_L]
    wr = kp[WRIST_R] if side == "right" else kp[WRIST_L]
    shoulder_lat = abs(np.dot(sh - kp[PELVIS], frame.side))
    wrist_lat = abs(np.dot(wr - kp[PELVIS], frame.side))
    threshold = shoulder_lat * (1.0 + RULA_ARM_OUT_TO_SIDE_RATIO * stature_scale_factor(stature_cm))
    return wrist_lat > threshold


def is_knee_at_floor(
    pose: Major25Pose, anthro: AnthropometryReport, side: str = "right",
) -> bool:
    """Per `standards/owas.md §4.3`: kneeling detection (OWAS legs = 6).

    Legacy uses 60 mm absolute knee-to-toe distance. We use a ratio of
    the worker's leg total length.
    """
    kp = pose.keypoints
    if side == "right":
        d = float(np.linalg.norm(kp[KNEE_R] - kp[BIG_TOE_R]))
        leg = anthro.leg_total_r
    else:
        d = float(np.linalg.norm(kp[KNEE_L] - kp[BIG_TOE_L]))
        leg = anthro.leg_total_l
    return d < OWAS_KNEE_GROUND_RATIO * max(leg, 1e-6)


def limb_length_consistency_ok(anthro: AnthropometryReport) -> bool:
    """Audit recommendation `audit §A.6`: detect bad pose estimates
    when limb lengths on left vs right differ wildly.

    Returns True if the pose looks anatomically consistent.
    """
    def asym(a, b):
        return abs(a - b) / max(a, b, 1e-6)
    return all(
        asym(a, b) < 0.30
        for a, b in [
            (anthro.upper_arm_l, anthro.upper_arm_r),
            (anthro.forearm_l, anthro.forearm_r),
            (anthro.thigh_l, anthro.thigh_r),
            (anthro.shank_l, anthro.shank_r),
        ]
    )
