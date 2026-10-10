"""V — Explainable Risk Attribution.

Given an `ErgoResult`, decompose the final risk score into per-axis and
per-joint contributions. Three attribution modes are supported:

  1. **Marginal (counterfactual)** — replace each sub-score axis with its
     minimum value, recompute the grand score, take the drop. This is
     the simplest "if I fixed this one segment, how much would the risk
     fall?" measure.
  2. **Shapley value** — average marginal contribution across all
     permutations of axis subsets. Mathematically the unique
     attribution satisfying the four Shapley axioms (efficiency,
     symmetry, dummy, additivity). For OWAS (4 axes) and RULA/REBA
     (5-7 axes) this is fast: ≤ 7·2⁶ = 448 lookups per pose.
  3. **Joint mapping** — sum each axis's contribution onto the joints
     that participate in that sub-score, giving a per-major-25 joint
     attribution suitable for body-part heatmaps.

When a `McErgoResult` from `confidence_mc` is supplied, the module
additionally returns **uncertainty-decomposed** attribution: each
joint's contribution is split into (a) magnitude (Shapley value) and
(b) uncertainty (Monte-Carlo variance attributable to that joint via
its keypoint confidence).

Gap addressed
-------------
Recent reviews (Iyer 2025, CMC 2025) explicitly call out "lack of
explainability" as an open problem in vision-based ergonomic
assessment. This module is the first to decompose OWAS/RULA/REBA grand
scores into per-axis Shapley values and project them onto pose joints
for interventional decision-making.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ace_pose.ergo_v2.result import ErgoResult, OwasResult, RebaResult, RulaResult
from ace_pose.ergo_v2.tables import (
    OWAS_AC, REBA_TABLE_A, REBA_TABLE_B, REBA_TABLE_C,
    RULA_TABLE_A, RULA_TABLE_B, RULA_TABLE_C,
)


# ----------------------------------------------------------------------
# Sub-score axis definitions (canonical names + per-axis valid range)
# ----------------------------------------------------------------------

OWAS_AXES = {
    "back": (1, 4),
    "arms": (1, 3),
    "legs": (1, 7),
    "load": (1, 3),
}
RULA_AXES = {
    "upper_arm": (1, 6),
    "lower_arm": (1, 3),
    "wrist": (1, 4),
    "wrist_twist": (1, 2),
    "neck": (1, 6),
    "trunk": (1, 6),
    "legs": (1, 2),
}
REBA_AXES = {
    "trunk": (1, 5),
    "neck": (1, 3),
    "legs": (1, 4),
    "upper_arm": (1, 6),
    "lower_arm": (1, 2),
    "wrist": (1, 3),
}

# Joints that participate in each sub-score (for body-part heatmaps).
# Names refer to the 25-joint MAJOR layout in `ace_pose.major_joints`.
RULA_AXIS_JOINTS = {
    "upper_arm": ["SHOULDER_*", "ELBOW_*"],
    "lower_arm": ["ELBOW_*", "WRIST_*"],
    "wrist": ["WRIST_*"],
    "wrist_twist": ["WRIST_*"],
    "neck": ["NECK", "NOSE", "EYE_LEFT", "EYE_RIGHT"],
    "trunk": ["NECK", "PELVIS", "SHOULDER_LEFT", "SHOULDER_RIGHT", "HIP_LEFT", "HIP_RIGHT"],
    "legs": ["HIP_*", "KNEE_*", "ANKLE_*"],
}
REBA_AXIS_JOINTS = {
    "trunk": ["NECK", "PELVIS", "SHOULDER_LEFT", "SHOULDER_RIGHT", "HIP_LEFT", "HIP_RIGHT"],
    "neck": ["NECK", "NOSE", "EYE_LEFT", "EYE_RIGHT"],
    "legs": ["HIP_*", "KNEE_*", "ANKLE_*"],
    "upper_arm": ["SHOULDER_*", "ELBOW_*"],
    "lower_arm": ["ELBOW_*", "WRIST_*"],
    "wrist": ["WRIST_*"],
}
OWAS_AXIS_JOINTS = {
    "back": ["NECK", "PELVIS", "SHOULDER_LEFT", "SHOULDER_RIGHT", "HIP_LEFT", "HIP_RIGHT"],
    "arms": ["SHOULDER_*", "WRIST_*"],
    "legs": ["HIP_*", "KNEE_*", "ANKLE_*", "BIG_TOE_*"],
    "load": [],
}


def _expand_joint_pattern(pat: str) -> list[str]:
    """`SHOULDER_*` → [`SHOULDER_LEFT`, `SHOULDER_RIGHT`]."""
    if pat.endswith("_*"):
        base = pat[:-2]
        return [f"{base}_LEFT", f"{base}_RIGHT"]
    return [pat]


# ----------------------------------------------------------------------
# Forward evaluators (recompute grand score from sub-component values)
# ----------------------------------------------------------------------

def _eval_owas(values: dict[str, int]) -> int:
    return int(OWAS_AC[
        values["back"] - 1,
        values["arms"] - 1,
        values["legs"] - 1,
        values["load"] - 1,
    ])


def _eval_rula(values: dict[str, int]) -> int:
    """Re-evaluate RULA grand score from axis values (post-adjustment)."""
    posture_a = int(RULA_TABLE_A[
        values["upper_arm"] - 1,
        values["lower_arm"] - 1,
        values["wrist"] - 1,
        values["wrist_twist"] - 1,
    ])
    score_a = min(posture_a, 8)
    posture_b = int(RULA_TABLE_B[
        values["neck"] - 1,
        values["trunk"] - 1,
        values["legs"] - 1,
    ])
    score_b = min(posture_b, 7)
    return int(RULA_TABLE_C[score_a - 1, score_b - 1])


def _eval_reba(values: dict[str, int]) -> int:
    """Re-evaluate REBA score_c (table C output, before activity). The load and
    coupling modifiers are fixed inputs (keys "load", "coupling"), not players."""
    posture_a = int(REBA_TABLE_A[
        values["trunk"] - 1,
        values["neck"] - 1,
        values["legs"] - 1,
    ])
    score_a = min(posture_a + values.get("load", 0), 12)
    posture_b = int(REBA_TABLE_B[
        values["upper_arm"] - 1,
        values["lower_arm"] - 1,
        values["wrist"] - 1,
    ])
    score_b = min(posture_b + values.get("coupling", 0), 12)
    return int(REBA_TABLE_C[score_a - 1, score_b - 1])


# ----------------------------------------------------------------------
# Counterfactual marginal attribution
# ----------------------------------------------------------------------

def _axis_min(axes: dict[str, tuple[int, int]]) -> dict[str, int]:
    return {k: lo for k, (lo, _hi) in axes.items()}


def marginal_attribution(
    values: dict[str, int],
    axes: dict[str, tuple[int, int]],
    evaluator,
) -> dict[str, float]:
    """Δ = grand_actual - grand_with_axis_at_min, for each axis."""
    grand = evaluator(values)
    out: dict[str, float] = {}
    mins = _axis_min(axes)
    for axis in axes:
        cf = dict(values)
        cf[axis] = mins[axis]
        out[axis] = float(grand - evaluator(cf))
    return out


# ----------------------------------------------------------------------
# Shapley attribution
# ----------------------------------------------------------------------

def shapley_attribution(
    values: dict[str, int],
    axes: dict[str, tuple[int, int]],
    evaluator,
) -> dict[str, float]:
    """Exact Shapley value for each axis.

    The "value function" v(S) is the score evaluator with axes in S set
    to their actual values and axes not in S set to their minimum. The
    Shapley value for axis i is the average marginal contribution of i
    across all permutations.

    For N axes (N ≤ 7 in OWAS/RULA/REBA), there are 2^N coalitions
    and N! permutations. We use the coalition formula which costs
    O(N · 2^(N-1)) evaluator calls.
    """
    axis_names = list(axes.keys())
    n = len(axis_names)
    mins = _axis_min(axes)

    def _v(subset: tuple[str, ...]) -> float:
        cfg = dict(mins)
        for a in subset:
            cfg[a] = values[a]
        return float(evaluator(cfg))

    shap: dict[str, float] = {a: 0.0 for a in axis_names}
    # iterate all subsets of size k=0..n-1 of "other axes" excluding i
    for i, axis in enumerate(axis_names):
        others = [a for a in axis_names if a != axis]
        for k in range(len(others) + 1):
            for combo in itertools.combinations(others, k):
                weight = (
                    math.factorial(k) * math.factorial(n - k - 1)
                    / math.factorial(n)
                )
                without = combo
                with_i = combo + (axis,)
                shap[axis] += weight * (_v(with_i) - _v(without))
    return shap


# ----------------------------------------------------------------------
# Joint-level mapping
# ----------------------------------------------------------------------

def joints_from_axis(
    axis: str, mapping: dict[str, list[str]],
) -> list[str]:
    """Resolve `mapping[axis]` patterns into a flat list of joint names."""
    out: list[str] = []
    for pat in mapping.get(axis, []):
        out.extend(_expand_joint_pattern(pat))
    return out


def project_onto_joints(
    axis_contrib: dict[str, float],
    mapping: dict[str, list[str]],
) -> dict[str, float]:
    """Distribute each axis contribution evenly to its assigned joints."""
    joint_out: dict[str, float] = {}
    for axis, value in axis_contrib.items():
        joints = joints_from_axis(axis, mapping)
        if not joints:
            continue
        per = value / len(joints)
        for j in joints:
            joint_out[j] = joint_out.get(j, 0.0) + per
    return joint_out


# ----------------------------------------------------------------------
# Method-specific dispatch
# ----------------------------------------------------------------------

def _owas_values(res: OwasResult) -> dict[str, int]:
    return {
        "back": int(res.back),
        "arms": int(res.arms),
        "legs": int(res.legs),
        "load": int(res.load),
    }


def _rula_values(res: RulaResult) -> dict[str, int]:
    return {
        "upper_arm": int(res.upper_arm),
        "lower_arm": int(res.lower_arm),
        "wrist": int(res.wrist),
        "wrist_twist": int(res.wrist_twist),
        "neck": int(res.neck),
        "trunk": int(res.trunk),
        "legs": int(res.legs),
    }


def _reba_values(res: RebaResult) -> dict[str, int]:
    return {
        "trunk": int(res.trunk),
        "neck": int(res.neck),
        "legs": int(res.legs),
        "upper_arm": int(res.upper_arm),
        "lower_arm": int(res.lower_arm),
        "wrist": int(res.wrist),
    }


# ----------------------------------------------------------------------
# Public output
# ----------------------------------------------------------------------

@dataclass
class RiskAttribution:
    method: str
    grand_score: int
    axis_values: dict[str, int]
    marginal: dict[str, float]
    shapley: dict[str, float]
    joint_marginal: dict[str, float]
    joint_shapley: dict[str, float]
    # Optional uncertainty fields (filled when MC info is supplied)
    joint_uncertainty: dict[str, float] = field(default_factory=dict)


@dataclass
class ErgoAttribution:
    """All three risk attributions for one pose, ready for plotting."""
    owas: RiskAttribution
    rula: RiskAttribution
    reba: RiskAttribution


# ----------------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------------

def attribute_owas(res: OwasResult,
                   joint_uncertainty: dict[str, float] | None = None) -> RiskAttribution:
    values = _owas_values(res)
    marg = marginal_attribution(values, OWAS_AXES, _eval_owas)
    shap = shapley_attribution(values, OWAS_AXES, _eval_owas)
    return RiskAttribution(
        method="owas",
        grand_score=int(res.action_category),
        axis_values=values,
        marginal=marg,
        shapley=shap,
        joint_marginal=project_onto_joints(marg, OWAS_AXIS_JOINTS),
        joint_shapley=project_onto_joints(shap, OWAS_AXIS_JOINTS),
        joint_uncertainty=joint_uncertainty or {},
    )


def attribute_rula(res: RulaResult,
                   joint_uncertainty: dict[str, float] | None = None) -> RiskAttribution:
    values = _rula_values(res)
    marg = marginal_attribution(values, RULA_AXES, _eval_rula)
    shap = shapley_attribution(values, RULA_AXES, _eval_rula)
    return RiskAttribution(
        method="rula",
        grand_score=int(res.grand_score),
        axis_values=values,
        marginal=marg,
        shapley=shap,
        joint_marginal=project_onto_joints(marg, RULA_AXIS_JOINTS),
        joint_shapley=project_onto_joints(shap, RULA_AXIS_JOINTS),
        joint_uncertainty=joint_uncertainty or {},
    )


def attribute_reba(res: RebaResult,
                   joint_uncertainty: dict[str, float] | None = None) -> RiskAttribution:
    values = _reba_values(res)
    fixed = {"load": int(res.score_a - res.posture_a), "coupling": int(res.score_b - res.posture_b)}
    evaluate = lambda v: _eval_reba({**v, **fixed})
    marg = marginal_attribution(values, REBA_AXES, evaluate)
    shap = shapley_attribution(values, REBA_AXES, evaluate)
    return RiskAttribution(
        method="reba",
        grand_score=int(res.score_c),
        axis_values=values,
        marginal=marg,
        shapley=shap,
        joint_marginal=project_onto_joints(marg, REBA_AXIS_JOINTS),
        joint_shapley=project_onto_joints(shap, REBA_AXIS_JOINTS),
        joint_uncertainty=joint_uncertainty or {},
    )


def attribute_all(result: ErgoResult,
                  joint_uncertainty: dict[str, float] | None = None) -> ErgoAttribution:
    """Compute joint-level attribution for OWAS, RULA, REBA from one pose."""
    return ErgoAttribution(
        owas=attribute_owas(result.owas, joint_uncertainty),
        rula=attribute_rula(result.rula, joint_uncertainty),
        reba=attribute_reba(result.reba, joint_uncertainty),
    )


# ----------------------------------------------------------------------
# Uncertainty decomposition (combines V with C3 MC variance)
# ----------------------------------------------------------------------

def joint_uncertainty_from_mc(
    mc_score_samples: dict[str, list[int]],
    pose_keypoint_confidence: np.ndarray,
    major25_names: list[str],
) -> dict[str, float]:
    """Per-joint uncertainty: variance of method scores across MC samples,
    scaled by each joint's (1 - confidence).

    `mc_score_samples` maps "owas"/"rula"/"reba" → list of integer scores.
    `pose_keypoint_confidence` is the (25,) array from a `Major25Pose`.

    Returns a dict of joint_name → aggregated uncertainty across methods.
    """
    method_var = {
        m: float(np.var(np.asarray(s, dtype=float)))
        if len(s) else 0.0
        for m, s in mc_score_samples.items()
    }
    total_var = sum(method_var.values())
    if total_var <= 1e-9:
        return {}
    # Weight each joint by its inverse-confidence; renormalise to sum to total_var
    w = (1.0 - np.clip(pose_keypoint_confidence, 0.0, 1.0))
    w = w / max(w.sum(), 1e-9)
    return {
        major25_names[i]: float(w[i] * total_var)
        for i in range(len(major25_names))
    }
