"""REBA scoring from a major-25 pose.

Implements `standards/reba.md §3, §11` (3-D angle → sub-score) plus §4, §6,
§8, §9 (lookup tables and activity score). Fixes the legacy bugs identified
in `standards/audit.md §C`:
  - tables A, B, C transcribed verbatim (no errors found in legacy code,
    but we re-verify by `tables.py` shape and bounds asserts)
  - activity score parameters are exposed (caller drives them; conservative
    default `activity_static = True` for single-frame inference)
  - coupling defaults to 1 (Fair), surfaced as a parameter
  - canonical neck/trunk thresholds (0 / 20 / 60 trunk; 0 / 20 neck)
"""
from __future__ import annotations

import numpy as np

from ace_pose.major_joints import Major25Pose
from .angles import UPRIGHT_TOL_DEG, is_sitting
from .angles import (
    WRIST_DEFAULT_REBA, build_body_frame, elbow_flexion_deg,
    knee_flexion_deg, legs_balanced, neck_flexion_deg, neck_lateral_deg,
    neck_twist_deg, trunk_flexion_deg, trunk_lateral_deg, trunk_twist_deg,
    upper_arm_abduction_deg, upper_arm_flexion_deg, wrist_flexion_deg,
)
from .result import RebaResult
from .tables import REBA_TABLE_A, REBA_TABLE_B, REBA_TABLE_C


# Canonical thresholds — `standards/reba.md §3`.
TRUNK_T = (0.0, 20.0, 60.0)
NECK_T = (20.0,)                 # 1: 0..20 flexion; 2: > 20 flexion OR any extension
UPPER_ARM_T = (-20.0, 20.0, 45.0, 90.0)
LOWER_ARM_T = (60.0, 100.0)
WRIST_T = (15.0,)
KNEE_ADJ_T = (30.0, 60.0)        # +1 if 30-60°, +2 if >60°
TWIST_LATERAL_THRESHOLD = 10.0


def _score_trunk(deg: float) -> int:
    """1: upright (~0°); 2: 0..20 flex or ext; 3: 20..60 flex OR >20 ext; 4: >60 flex."""
    if abs(deg) <= UPRIGHT_TOL_DEG:     # upright, within measurement tolerance
        return 1
    if deg < 0:  # any extension
        return 2 if abs(deg) <= 20.0 else 3
    if deg <= TRUNK_T[1]:
        return 2
    if deg <= TRUNK_T[2]:
        return 3
    return 4


def _score_neck(deg: float) -> int:
    """1 if 0..20 flexion; 2 if >20 flexion OR extension beyond tolerance."""
    if -UPRIGHT_TOL_DEG <= deg <= NECK_T[0]:
        return 1
    return 2


def _score_legs(pose: Major25Pose, frame, supported: bool | None) -> tuple[int, list[str]]:
    """1: bilateral / walking / sitting; 2: unilateral / unstable.
    Adjustment: +1 if max knee flex 30-60°, +2 if >60°. Skipped when sitting."""
    notes: list[str] = []
    bilateral = legs_balanced(pose, frame) if supported is None else supported
    base = 1 if bilateral else 2
    if is_sitting(pose, frame):
        notes.append("legs: sitting, knee adjustment not applied")
        return base, notes
    knee_max = max(knee_flexion_deg(pose, "left"), knee_flexion_deg(pose, "right"))
    if knee_max > KNEE_ADJ_T[1]:
        base += 2
        notes.append(f"legs: knee flex {knee_max:.0f}° > 60° → +2")
    elif knee_max > KNEE_ADJ_T[0]:
        base += 1
        notes.append(f"legs: knee flex {knee_max:.0f}° > 30° → +1")
    return min(base, 4), notes


def _score_upper_arm(deg: float) -> int:
    if UPPER_ARM_T[0] <= deg <= UPPER_ARM_T[1]:
        return 1
    if deg < UPPER_ARM_T[0] or (UPPER_ARM_T[1] < deg <= UPPER_ARM_T[2]):
        return 2
    if UPPER_ARM_T[2] < deg <= UPPER_ARM_T[3]:
        return 3
    return 4


def _score_lower_arm(deg: float) -> int:
    return 1 if LOWER_ARM_T[0] <= deg <= LOWER_ARM_T[1] else 2


def _score_wrist(deg: float | None, *, default: int = WRIST_DEFAULT_REBA) -> int:
    if deg is None:
        return default
    return 1 if abs(deg) <= WRIST_T[0] else 2


def _load_score(load_kg: float, shock: bool) -> int:
    if load_kg < 5.0:
        base = 0
    elif load_kg <= 10.0:
        base = 1
    else:
        base = 2
    return base + (1 if shock else 0)


def _activity_score(static: bool, repetitive: bool, rapid: bool) -> int:
    return int(static) + int(repetitive) + int(rapid)


def _score_one_side(
    pose: Major25Pose,
    frame,
    side: str,
    *,
    world_up,
    load_kg: float,
    shock: bool,
    coupling: int,
    activity_static: bool,
    activity_repetitive: bool,
    activity_rapid: bool,
    legs_supported: bool | None,
    shoulder_raised: bool,
    upper_arm_supported: bool,
    wrist_deviation: bool,
    wrist_score_override: int | None,
) -> RebaResult:
    upper_deg = upper_arm_flexion_deg(pose, frame, side=side)
    abduction = upper_arm_abduction_deg(pose, frame, side=side)
    elbow_deg = elbow_flexion_deg(pose, side=side)
    wrist_deg = wrist_flexion_deg(pose, side=side)

    # ---- Group A: trunk, neck, legs ----
    trunk_deg = trunk_flexion_deg(pose, frame, world_up=world_up)
    trunk_twist = trunk_twist_deg(pose, frame)
    trunk_lat = trunk_lateral_deg(pose, frame, world_up=world_up)
    neck_deg = neck_flexion_deg(pose, frame, world_up=world_up)
    neck_twist = neck_twist_deg(pose, frame)
    neck_lat = neck_lateral_deg(pose, frame, world_up=world_up)

    tr = _score_trunk(trunk_deg)
    if trunk_twist > TWIST_LATERAL_THRESHOLD or trunk_lat > TWIST_LATERAL_THRESHOLD:
        tr += 1
    tr = max(1, min(5, tr))

    nk = _score_neck(neck_deg)
    if neck_twist > TWIST_LATERAL_THRESHOLD or neck_lat > TWIST_LATERAL_THRESHOLD:
        nk += 1
    nk = max(1, min(3, nk))

    lg, leg_notes = _score_legs(pose, frame, supported=legs_supported)

    posture_a = int(REBA_TABLE_A[tr - 1, nk - 1, lg - 1])
    load = _load_score(load_kg, shock)
    score_a = min(posture_a + load, 12)

    # ---- Group B: upper arm, lower arm, wrist ----
    ua = _score_upper_arm(upper_deg)
    if shoulder_raised: ua += 1
    if abduction > 30.0: ua += 1
    if upper_arm_supported: ua -= 1
    ua = max(1, min(6, ua))

    la = _score_lower_arm(elbow_deg)

    wr = wrist_score_override if wrist_score_override is not None else _score_wrist(wrist_deg)
    if wrist_deviation: wr += 1
    wr = max(1, min(3, wr))

    posture_b = int(REBA_TABLE_B[ua - 1, la - 1, wr - 1])
    score_b = min(posture_b + max(0, min(coupling, 3)), 12)

    # ---- Table C + Activity ----
    score_c = int(REBA_TABLE_C[score_a - 1, score_b - 1])
    activity = _activity_score(activity_static, activity_repetitive, activity_rapid)
    grand = min(score_c + activity, 15)

    if grand == 1:
        action = 0
    elif grand <= 3:
        action = 1
    elif grand <= 7:
        action = 2
    elif grand <= 10:
        action = 3
    else:
        action = 4

    return RebaResult(
        trunk=tr, neck=nk, legs=lg,
        posture_a=posture_a, score_a=score_a,
        upper_arm=ua, lower_arm=la, wrist=wr,
        posture_b=posture_b, score_b=score_b,
        score_c=score_c, activity_score=activity, grand_score=grand,
        action_level=action,
        trunk_deg=float(trunk_deg), neck_deg=float(neck_deg),
        upper_arm_deg=float(upper_deg), lower_arm_deg=float(elbow_deg),
        wrist_deg=float(wrist_deg) if wrist_deg is not None else float("nan"),
        knee_deg=float(max(knee_flexion_deg(pose, "left"),
                           knee_flexion_deg(pose, "right"))),
        load_kg=float(load_kg), coupling=int(coupling),
        activity_static=bool(activity_static),
        activity_repetitive=bool(activity_repetitive),
        activity_rapid=bool(activity_rapid),
        side=side,
        notes=[
            f"trunk: {trunk_deg:+.1f}° (twist {trunk_twist:.1f}°, lat {trunk_lat:.1f}°) → {tr}",
            f"neck: {neck_deg:+.1f}° (twist {neck_twist:.1f}°, lat {neck_lat:.1f}°) → {nk}",
        ] + leg_notes + [
            f"legs: → {lg}",
            f"score_a = posture_a {posture_a} + load {load} → {score_a}",
            f"upper_arm: {upper_deg:+.1f}° (abd {abduction:.1f}°) → {ua}",
            f"lower_arm: {elbow_deg:.1f}° → {la}",
            f"wrist: {wrist_deg} → {wr} (default-mode={wrist_deg is None})",
            f"score_b = posture_b {posture_b} + coupling {coupling} → {score_b}",
            f"score_c = {score_c} + activity {activity} → grand {grand} (action {action})",
        ],
    )


def score_reba(
    pose: Major25Pose,
    *,
    side: str = "max",
    world_up: np.ndarray | None = None,
    load_kg: float = 0.0,
    shock: bool = False,
    coupling: int = 1,                # default Fair, per audit recommendation
    activity_static: bool = True,     # conservative default for single-frame
    activity_repetitive: bool = False,
    activity_rapid: bool = False,
    legs_supported: bool | None = None,
    shoulder_raised: bool = False,
    upper_arm_supported: bool = False,
    wrist_deviation: bool = False,
    wrist_score_override: int | None = None,
) -> RebaResult:
    """Score a single pose against the REBA standard.

    Defaults follow `standards/reba.md §13` recommendations:
      - coupling = 1 (Fair) rather than 0 (Good)
      - activity_static = True for single-frame inference
      - wrist defaults to 1 with no finger keypoints; override via
        `wrist_score_override` when hand pose is available.
    """
    frame = build_body_frame(pose)
    if side in ("left", "right"):
        return _score_one_side(
            pose, frame, side, world_up=world_up,
            load_kg=load_kg, shock=shock, coupling=coupling,
            activity_static=activity_static,
            activity_repetitive=activity_repetitive,
            activity_rapid=activity_rapid,
            legs_supported=legs_supported,
            shoulder_raised=shoulder_raised,
            upper_arm_supported=upper_arm_supported,
            wrist_deviation=wrist_deviation,
            wrist_score_override=wrist_score_override,
        )
    left = _score_one_side(
        pose, frame, "left", world_up=world_up,
        load_kg=load_kg, shock=shock, coupling=coupling,
        activity_static=activity_static,
        activity_repetitive=activity_repetitive,
        activity_rapid=activity_rapid,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
        wrist_score_override=wrist_score_override,
    )
    right = _score_one_side(
        pose, frame, "right", world_up=world_up,
        load_kg=load_kg, shock=shock, coupling=coupling,
        activity_static=activity_static,
        activity_repetitive=activity_repetitive,
        activity_rapid=activity_rapid,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
        wrist_score_override=wrist_score_override,
    )
    return right if right.grand_score >= left.grand_score else left
