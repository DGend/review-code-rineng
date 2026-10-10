"""OWAS scoring from a major-25 pose.

Implements `standards/owas.md §4` (3-D angle → digit) and §3 (lookup).
Fixes the three legacy bugs identified in `standards/audit.md §A`:
  - bent + lateral-only correctly yields Back = 4
  - 150° canonical knee threshold (not 160°)
  - deep squats are NOT misclassified as sitting (sit detection uses hip
    flexion + low pelvis, not just knee-angle < 90°)
"""
from __future__ import annotations

import numpy as np

from ace_pose.major_joints import Major25Pose
from .angles import (
    arms_above_shoulder, build_body_frame, is_kneeling, is_sitting,
    knee_flexion_deg, legs_balanced, trunk_flexion_deg, trunk_lateral_deg,
    trunk_twist_deg,
)
from .result import OwasResult
from .tables import OWAS_AC


# Canonical thresholds — `standards/owas.md §2`.
OWAS_BACK_FLEX_THRESHOLD = 20.0      # ° sagittal
OWAS_BACK_TWIST_THRESHOLD = 20.0     # ° axial or lateral
OWAS_KNEE_BENT_THRESHOLD = 30.0      # ° flexion (≡ 150° interior angle)


def _score_back(flex: float, twist: float, lateral: float) -> tuple[int, list[str]]:
    """Per `standards/owas.md §2.1` and audit fix §A.3.

    Back = 1 + int(bent) + 2 * int(twisted), where:
      bent     = |sagittal_flex| > 20°
      twisted  = max(|axial_twist|, |lateral_bend|) > 20°
    """
    notes: list[str] = []
    bent = abs(flex) > OWAS_BACK_FLEX_THRESHOLD
    twisted = max(abs(twist), abs(lateral)) > OWAS_BACK_TWIST_THRESHOLD
    code = 1 + int(bent) + 2 * int(twisted)
    notes.append(f"back: flex={flex:+.1f}°, twist={twist:.1f}°, lateral={lateral:.1f}° → {code}")
    return code, notes


def _score_arms(arms_above: int) -> int:
    """Per `standards/owas.md §2.2`: 0 above → 1, 1 above → 2, 2 above → 3."""
    return 1 + min(arms_above, 2)


def _score_legs(
    pose: Major25Pose,
    knee_l: float,
    knee_r: float,
    walking: bool,
    kneeling: bool,
    sitting: bool,
    balanced: bool,
) -> tuple[int, list[str]]:
    """Per `standards/owas.md §2.3` and §4.3 priority pipeline.

    Order: walking → kneeling → sitting → bent/straight × balanced/single.
    """
    notes: list[str] = []
    bent_l = knee_l > OWAS_KNEE_BENT_THRESHOLD
    bent_r = knee_r > OWAS_KNEE_BENT_THRESHOLD

    if walking:
        notes.append("legs: walking flag → 7")
        return 7, notes
    if kneeling:
        notes.append("legs: kneeling → 6")
        return 6, notes
    if sitting:
        notes.append("legs: sitting → 1")
        return 1, notes

    # standing posture
    if not bent_l and not bent_r:
        code = 3 if not balanced else 2
        notes.append(f"legs: both straight → {code} ({'unbalanced' if not balanced else 'balanced'})")
        return code, notes
    if bent_l and bent_r:
        code = 5 if not balanced else 4
        notes.append(f"legs: both bent → {code} ({'unbalanced' if not balanced else 'balanced'})")
        return code, notes
    # exactly one bent: codes 3 and 5 both mean the weight is on one leg, which the
    # balanced (both-feet) assumption rules out, so this is standing on both legs
    code = 2 if balanced else 3
    notes.append(f"legs: one knee bent (L={knee_l:.0f}°, R={knee_r:.0f}°) → {code}")
    return code, notes


def score_owas(
    pose: Major25Pose,
    *,
    load_kg: float = 0.0,
    walking: bool = False,
    world_up: np.ndarray | None = None,
) -> OwasResult:
    """Score a single pose against the OWAS standard.

    Args:
        pose: Major-25 keypoint pose (output of
            `ace_pose.major_joints.coco_wholebody_to_major25`).
        load_kg: handled load in kilograms. Per standard:
            <10 → 1, 10-20 → 2, >20 → 3.
        walking: caller-supplied flag (cannot be derived from a single frame).
        world_up: world up-axis as a unit vector. Defaults to body-up if
            None (rough fallback for stand-alone inference).

    Returns:
        OwasResult with action category (1..4) and traceability detail.
    """
    frame = build_body_frame(pose)
    flex = trunk_flexion_deg(pose, frame, world_up=world_up)
    twist = trunk_twist_deg(pose, frame)
    lateral = trunk_lateral_deg(pose, frame, world_up=world_up)

    back_code, back_notes = _score_back(flex, twist, lateral)
    arms_above = arms_above_shoulder(pose, frame)
    arms_code = _score_arms(arms_above)

    knee_l = knee_flexion_deg(pose, "left")
    knee_r = knee_flexion_deg(pose, "right")
    balanced = legs_balanced(pose, frame)
    kneeling = is_kneeling(pose, frame)
    sitting = is_sitting(pose, frame)

    legs_code, legs_notes = _score_legs(
        pose, knee_l, knee_r, walking, kneeling, sitting, balanced,
    )

    # load mapping
    if load_kg < 10.0:
        load_code = 1
    elif load_kg <= 20.0:
        load_code = 2
    else:
        load_code = 3

    ac = int(OWAS_AC[back_code - 1, arms_code - 1, legs_code - 1, load_code - 1])

    return OwasResult(
        back=back_code,
        arms=arms_code,
        legs=legs_code,
        load=load_code,
        action_category=ac,
        trunk_flex_deg=float(flex),
        trunk_twist_deg=float(twist),
        trunk_lateral_deg=float(lateral),
        arms_above_count=arms_above,
        knee_left_deg=float(knee_l),
        knee_right_deg=float(knee_r),
        notes=back_notes + [f"arms: {arms_above} above shoulder → {arms_code}"] + legs_notes + [
            f"load: {load_kg} kg → {load_code}",
            f"action_category: {ac}",
        ],
    )
