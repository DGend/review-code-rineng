"""RULA scoring from a major-25 pose.

Implements `standards/rula.md §3, §9` (3-D angle → sub-score) plus §4, §5,
§7 (lookup tables). Fixes the legacy bugs identified in
`standards/audit.md §B`:
  - Table A transcribed verbatim from Cornell PDF (26 cells corrected)
  - canonical thresholds (0° / 20° / 60° trunk; 0° / 10° / 20° neck;
    -20° / 20° / 45° / 90° upper-arm)
  - wrist defaults exposed as parameters; not hard-coded to 1
"""
from __future__ import annotations

import numpy as np

from ace_pose.major_joints import Major25Pose
from .angles import (
    WRIST_DEFAULT_RULA, WRIST_TWIST_DEFAULT_RULA,
    UPRIGHT_TOL_DEG, build_body_frame, elbow_flexion_deg, forearm_across_midline,
    forearm_out_to_side, knee_flexion_deg, legs_balanced,
    neck_flexion_deg, neck_lateral_deg, neck_twist_deg,
    trunk_flexion_deg, trunk_lateral_deg, trunk_twist_deg,
    upper_arm_abduction_deg, upper_arm_flexion_deg, wrist_flexion_deg,
)
from .result import RulaResult
from .tables import RULA_TABLE_A, RULA_TABLE_B, RULA_TABLE_C


# Canonical thresholds — `standards/rula.md §3`.
UPPER_ARM_T = (-20.0, 20.0, 45.0, 90.0)   # 1: -20..20, 2: 20..45 or <-20, 3: 45..90, 4: >90
LOWER_ARM_T = (60.0, 100.0)               # 1: 60..100 (open intervals match Cornell)
WRIST_T = (15.0,)                         # 1: 0, 2: 0..15, 3: >15
NECK_T = (10.0, 20.0)                     # 1: 0..10, 2: 10..20, 3: >20, 4: any extension
TRUNK_T = (0.0, 20.0, 60.0)               # 1: 0, 2: 0..20, 3: 20..60, 4: >60
TWIST_LATERAL_THRESHOLD = 10.0            # +1 trunk/neck adjustment


def _score_upper_arm(deg: float) -> int:
    """deg = sagittal flexion; positive forward."""
    if UPPER_ARM_T[0] <= deg <= UPPER_ARM_T[1]:
        return 1
    if deg < UPPER_ARM_T[0] or (UPPER_ARM_T[1] < deg <= UPPER_ARM_T[2]):
        return 2
    if UPPER_ARM_T[2] < deg <= UPPER_ARM_T[3]:
        return 3
    return 4  # deg > 90


def _score_lower_arm(deg: float) -> int:
    """Elbow flexion ∈ [0, 180]. Score 1 inside open (60, 100); else 2."""
    if LOWER_ARM_T[0] < deg < LOWER_ARM_T[1]:
        return 1
    return 2


def _score_wrist(deg: float | None, *, default: int = WRIST_DEFAULT_RULA) -> int:
    if deg is None:
        return default
    a = abs(deg)
    if a < 1e-6:
        return 1
    if a <= WRIST_T[0]:
        return 2
    return 3


def _score_neck(deg: float) -> int:
    """deg = signed flexion; positive forward, negative extension."""
    if deg < -UPRIGHT_TOL_DEG:
        return 4  # extension beyond measurement tolerance
    deg = max(deg, 0.0)
    if deg <= NECK_T[0]:
        return 1
    if deg <= NECK_T[1]:
        return 2
    return 3


def _score_trunk(deg: float) -> int:
    """deg = signed flexion. Score 1 only when 0; (0, 20] → 2; (20, 60] → 3; >60 → 4."""
    flex = abs(deg)
    if flex <= TRUNK_T[0] + UPRIGHT_TOL_DEG:   # "upright", within measurement tolerance
        return 1
    if flex <= TRUNK_T[1]:
        return 2
    if flex <= TRUNK_T[2]:
        return 3
    return 4


def _score_legs(pose: Major25Pose, frame, supported: bool | None = None) -> int:
    """1 if supported + balanced; else 2."""
    if supported is False:
        return 2
    return 1 if legs_balanced(pose, frame) else 2


def _force_load_score(load_kg: float, static_or_repeat: bool, shock: bool) -> int:
    """`standards/rula.md §6.2`."""
    if shock or (load_kg > 10 and static_or_repeat):
        return 3
    if load_kg > 10:
        return 2
    if 2 <= load_kg <= 10:
        return 2 if static_or_repeat else 1
    return 0


def _muscle_use_score(static_minute: bool, repetitive_4_per_min: bool) -> int:
    return 1 if (static_minute or repetitive_4_per_min) else 0


def _score_one_side(
    pose: Major25Pose,
    frame,
    side: str,
    *,
    world_up,
    load_kg: float,
    static_minute: bool,
    repetitive_4_per_min: bool,
    load_repeated: bool,
    shock: bool,
    legs_supported: bool | None,
    shoulder_raised: bool,
    upper_arm_supported: bool,
    wrist_deviation: bool,
    wrist_twist_at_end: bool,
    wrist_score_override: int | None,
    wrist_twist_score_override: int | None,
) -> RulaResult:
    upper_deg = upper_arm_flexion_deg(pose, frame, side=side)
    abduction = upper_arm_abduction_deg(pose, frame, side=side)
    elbow_deg = elbow_flexion_deg(pose, side=side)
    wrist_deg = wrist_flexion_deg(pose, side=side)  # None without finger keypoints

    # --- Group A ---
    ua = _score_upper_arm(upper_deg)
    if shoulder_raised: ua += 1
    if abduction > 30.0: ua += 1
    if upper_arm_supported: ua -= 1
    ua = max(1, min(6, ua))

    la = _score_lower_arm(elbow_deg)
    # "working across the midline or out to the side": +1 at most
    if forearm_across_midline(pose, frame, side=side) or forearm_out_to_side(pose, frame, side=side):
        la += 1
    la = max(1, min(3, la))

    wr = wrist_score_override if wrist_score_override is not None else _score_wrist(wrist_deg)
    if wrist_deviation: wr += 1
    wr = max(1, min(4, wr))

    wt = (wrist_twist_score_override if wrist_twist_score_override is not None
          else (2 if wrist_twist_at_end else WRIST_TWIST_DEFAULT_RULA))

    posture_a = int(RULA_TABLE_A[ua - 1, la - 1, wr - 1, wt - 1])
    muscle_a = _muscle_use_score(static_minute, repetitive_4_per_min)
    load_a = _force_load_score(load_kg, static_minute or repetitive_4_per_min or load_repeated, shock)
    score_a_raw = posture_a + muscle_a + load_a
    score_a_clamped = min(score_a_raw, 8)

    # --- Group B ---
    trunk_deg = trunk_flexion_deg(pose, frame, world_up=world_up)
    trunk_twist = trunk_twist_deg(pose, frame)
    trunk_lateral = trunk_lateral_deg(pose, frame, world_up=world_up)
    neck_deg = neck_flexion_deg(pose, frame, world_up=world_up)
    neck_twist = neck_twist_deg(pose, frame)
    neck_lateral = neck_lateral_deg(pose, frame, world_up=world_up)

    nk = _score_neck(neck_deg)
    if neck_twist > TWIST_LATERAL_THRESHOLD: nk += 1
    if neck_lateral > TWIST_LATERAL_THRESHOLD: nk += 1
    nk = max(1, min(6, nk))

    tr = _score_trunk(trunk_deg)
    if trunk_twist > TWIST_LATERAL_THRESHOLD: tr += 1
    if trunk_lateral > TWIST_LATERAL_THRESHOLD: tr += 1
    tr = max(1, min(6, tr))

    lg = _score_legs(pose, frame, supported=legs_supported)

    posture_b = int(RULA_TABLE_B[nk - 1, tr - 1, lg - 1])
    muscle_b = _muscle_use_score(static_minute, repetitive_4_per_min)
    load_b = _force_load_score(load_kg, static_minute or repetitive_4_per_min or load_repeated, shock)
    score_b_raw = posture_b + muscle_b + load_b
    score_b_clamped = min(score_b_raw, 7)

    grand = int(RULA_TABLE_C[score_a_clamped - 1, score_b_clamped - 1])
    action = (1 if grand <= 2 else 2 if grand <= 4 else 3 if grand <= 6 else 4)

    return RulaResult(
        upper_arm=ua, lower_arm=la, wrist=wr, wrist_twist=wt,
        posture_a=posture_a, score_a=score_a_clamped,
        neck=nk, trunk=tr, legs=lg,
        posture_b=posture_b, score_b=score_b_clamped,
        grand_score=grand, action_level=action,
        upper_arm_deg=float(upper_deg),
        lower_arm_deg=float(elbow_deg),
        wrist_deg=float(wrist_deg) if wrist_deg is not None else float("nan"),
        neck_deg=float(neck_deg),
        trunk_deg=float(trunk_deg),
        trunk_twist_deg=float(trunk_twist),
        neck_twist_deg=float(neck_twist),
        muscle_use_a=bool(muscle_a),
        muscle_use_b=bool(muscle_b),
        load_a=int(load_a),
        load_b=int(load_b),
        side=side,
        notes=[
            f"upper_arm: {upper_deg:+.1f}° (abd {abduction:.1f}°) → {ua}",
            f"lower_arm: {elbow_deg:.1f}° → {la}",
            f"wrist: {wrist_deg} → {wr} (twist {wt}); default-mode={wrist_deg is None}",
            f"neck: {neck_deg:+.1f}° (twist {neck_twist:.1f}°, lateral {neck_lateral:.1f}°) → {nk}",
            f"trunk: {trunk_deg:+.1f}° (twist {trunk_twist:.1f}°, lateral {trunk_lateral:.1f}°) → {tr}",
            f"legs: → {lg}",
            f"score_a={score_a_clamped} (raw {score_a_raw}, muscle {muscle_a}, load {load_a})",
            f"score_b={score_b_clamped} (raw {score_b_raw}, muscle {muscle_b}, load {load_b})",
            f"grand={grand} → action_level {action}",
        ],
    )


def score_rula(
    pose: Major25Pose,
    *,
    side: str = "max",
    world_up: np.ndarray | None = None,
    load_kg: float = 0.0,
    static_minute: bool = False,
    repetitive_4_per_min: bool = False,
    load_repeated: bool = False,
    shock: bool = False,
    legs_supported: bool | None = None,
    shoulder_raised: bool = False,
    upper_arm_supported: bool = False,
    wrist_deviation: bool = False,
    wrist_twist_at_end: bool = False,
    wrist_score_override: int | None = None,
    wrist_twist_score_override: int | None = None,
) -> RulaResult:
    """Score a single pose against the RULA standard.

    Args:
        pose: Major-25 keypoint pose.
        side: which side to score. "left" / "right" / "max" (worse side).
        world_up: world up-axis. Defaults to body-up fallback.
        load_kg: handled load in kg.
        static_minute, repetitive_4_per_min: per-segment muscle-use flags.
        load_repeated: the load is handled repeatedly (force/load score), without
            implying a held or repeated posture (muscle use).
        shock: shock or sudden force flag.
        legs_supported: tri-state (True / False / None=auto).
        shoulder_raised, upper_arm_supported, wrist_deviation,
            wrist_twist_at_end: optional caller-supplied flags.
        wrist_score_override, wrist_twist_score_override: caller can set
            the wrist sub-scores explicitly when finger pose is available.

    Returns:
        RulaResult for the requested side (or the worse of L/R if side="max").
    """
    frame = build_body_frame(pose)
    if side in ("left", "right"):
        return _score_one_side(
            pose, frame, side, world_up=world_up,
            load_kg=load_kg, static_minute=static_minute,
            repetitive_4_per_min=repetitive_4_per_min, load_repeated=load_repeated, shock=shock,
            legs_supported=legs_supported,
            shoulder_raised=shoulder_raised,
            upper_arm_supported=upper_arm_supported,
            wrist_deviation=wrist_deviation,
            wrist_twist_at_end=wrist_twist_at_end,
            wrist_score_override=wrist_score_override,
            wrist_twist_score_override=wrist_twist_score_override,
        )
    left = _score_one_side(
        pose, frame, "left", world_up=world_up,
        load_kg=load_kg, static_minute=static_minute,
        repetitive_4_per_min=repetitive_4_per_min, load_repeated=load_repeated, shock=shock,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
        wrist_twist_at_end=wrist_twist_at_end,
        wrist_score_override=wrist_score_override,
        wrist_twist_score_override=wrist_twist_score_override,
    )
    right = _score_one_side(
        pose, frame, "right", world_up=world_up,
        load_kg=load_kg, static_minute=static_minute,
        repetitive_4_per_min=repetitive_4_per_min, load_repeated=load_repeated, shock=shock,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
        wrist_twist_at_end=wrist_twist_at_end,
        wrist_score_override=wrist_score_override,
        wrist_twist_score_override=wrist_twist_score_override,
    )
    return right if right.grand_score >= left.grand_score else left
