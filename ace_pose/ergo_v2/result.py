"""Unified result schema for ergo_v2 scorers.

Every scorer (OWAS, RULA, REBA) returns a dataclass instance carrying:
  - the final action level / risk score
  - the per-segment sub-scores
  - the underlying raw angles (for downstream confidence-MC and XAI)
  - the input flags used (anthropometric/process/context)
  - traceability metadata for explainable risk attribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OwasResult:
    back: int                   # 1..4
    arms: int                   # 1..3
    legs: int                   # 1..7
    load: int                   # 1..3
    action_category: int        # 1..4
    # detail
    trunk_flex_deg: float = 0.0
    trunk_twist_deg: float = 0.0
    trunk_lateral_deg: float = 0.0
    arms_above_count: int = 0
    knee_left_deg: float = 0.0
    knee_right_deg: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class RulaResult:
    # Group A
    upper_arm: int
    lower_arm: int
    wrist: int
    wrist_twist: int
    posture_a: int              # before muscle + force
    score_a: int                # after muscle + force, clamped 1..8 for table-C
    # Group B
    neck: int
    trunk: int
    legs: int
    posture_b: int
    score_b: int                # clamped 1..7 for table-C
    # final
    grand_score: int            # 1..7
    action_level: int           # 1..4
    # detail
    upper_arm_deg: float = 0.0
    lower_arm_deg: float = 0.0
    wrist_deg: float = 0.0
    neck_deg: float = 0.0
    trunk_deg: float = 0.0
    trunk_twist_deg: float = 0.0
    neck_twist_deg: float = 0.0
    muscle_use_a: bool = False
    muscle_use_b: bool = False
    load_a: int = 0
    load_b: int = 0
    side: str = "right"         # which side scored ("left" or "right" or "max")
    notes: list[str] = field(default_factory=list)


@dataclass
class RebaResult:
    # Group A (trunk + neck + legs)
    trunk: int
    neck: int
    legs: int
    posture_a: int
    score_a: int                # after load
    # Group B (upper + lower + wrist)
    upper_arm: int
    lower_arm: int
    wrist: int
    posture_b: int
    score_b: int                # after coupling
    # combined
    score_c: int                # table C output (1..12)
    activity_score: int         # 0..3
    grand_score: int            # 1..15
    action_level: int           # 0..4
    # detail
    trunk_deg: float = 0.0
    neck_deg: float = 0.0
    upper_arm_deg: float = 0.0
    lower_arm_deg: float = 0.0
    wrist_deg: float = 0.0
    knee_deg: float = 0.0
    load_kg: float = 0.0
    coupling: int = 1
    activity_static: bool = False
    activity_repetitive: bool = False
    activity_rapid: bool = False
    side: str = "right"
    notes: list[str] = field(default_factory=list)


@dataclass
class ErgoResult:
    """Combined ergo assessment from all three methods on one pose."""
    owas: OwasResult
    rula: RulaResult
    reba: RebaResult
    # carries (left, right) tuples if scored bilaterally
    rula_left: RulaResult | None = None
    rula_right: RulaResult | None = None
    reba_left: RebaResult | None = None
    reba_right: RebaResult | None = None
    # metadata
    process_id: int | None = None
    process_label: str | None = None
    domain: str | None = None       # "manufacturing" | "eldercare" | ...
    extras: dict[str, Any] = field(default_factory=dict)
