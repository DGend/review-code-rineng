"""X — Biomechanical inverse dynamics on a single static pose.

Computes STATIC joint moments (gravity torques only) at the major
articulations of a worker, given a 3-D pose. Uses canonical
segment-mass and centre-of-mass distributions from de Leva 1996 /
Winter 2009, scaled to the worker's body mass.

Inputs
------
* `pose` — `Major25Pose` (the 25-joint output of
  `coco_wholebody_to_major25`). Coordinates may be in any consistent
  units; downstream computations scale by `body_mass_kg` and a
  user-supplied `length_scale_m` so the result is in Newton-metres.
* `body_mass_kg` — total worker mass. Defaults to 70 kg (cohort mean).
* `length_scale_m` — conversion factor from pose units to metres.
  Pass 1.0 if pose is already metric (e.g. MoGe-2 output); pass
  `stature_m / measured_stature_in_pose_units` otherwise.

Outputs
-------
A `BiomechResult` dataclass carrying:
  - per-joint static gravity torque (N·m)
  - per-joint dimensionless torque (Nm / (m × g × M)) for cross-subject
    comparison
  - per-segment force, moment arm, and direction-cosine details for
    explainability (also feeds the V XAI module)

Limitations
-----------
* Single-frame STATIC analysis only: no inertia, no joint-acceleration
  contributions. Suitable for postures held > ~0.5 s.
* External loads (hand-held objects, foot reaction) are not modelled
  unless explicitly supplied via `external_loads`.
* Friction and muscle co-contraction are not modelled — the output is
  a NET external moment that the joint structure must resist.

Reference
---------
de Leva P. "Adjustments to Zatsiorsky-Seluyanov's segment inertia
parameters." J Biomech 29(9):1223-1230, 1996.
Winter DA. "Biomechanics and Motor Control of Human Movement", 4th ed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ace_pose.major_joints import (
    Major25Pose, ANKLE_L, ANKLE_R, ELBOW_L, ELBOW_R, HIP_L, HIP_R,
    KNEE_L, KNEE_R, NECK, NOSE, PELVIS, SHOULDER_L, SHOULDER_R,
    WRIST_L, WRIST_R,
)


# ----------------------------------------------------------------------
# Segment parameters (de Leva 1996, adult male/female averaged)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    name: str
    mass_fraction: float          # of total body mass
    com_fraction: float           # along proximal→distal axis (0..1)
    proximal: int                 # major-25 joint index
    distal: int                   # major-25 joint index


# Mass fractions sum across all segments (head, trunk, both arms,
# both legs) ≈ 1.0 (foot mass folded into shank for simplicity).
SEGMENTS: tuple[Segment, ...] = (
    Segment("head_neck", 0.0681, 0.55, NECK, NOSE),
    Segment("trunk",     0.4346, 0.50, NECK, PELVIS),
    Segment("upper_arm_l", 0.0271, 0.58, SHOULDER_L, ELBOW_L),
    Segment("upper_arm_r", 0.0271, 0.58, SHOULDER_R, ELBOW_R),
    Segment("forearm_l",  0.0162, 0.46, ELBOW_L, WRIST_L),
    Segment("forearm_r",  0.0162, 0.46, ELBOW_R, WRIST_R),
    Segment("hand_l",     0.0061, 0.50, WRIST_L, WRIST_L),  # CoM at wrist
    Segment("hand_r",     0.0061, 0.50, WRIST_R, WRIST_R),
    Segment("thigh_l",    0.1416, 0.41, HIP_L, KNEE_L),
    Segment("thigh_r",    0.1416, 0.41, HIP_R, KNEE_R),
    Segment("shank_foot_l", 0.0570, 0.45, KNEE_L, ANKLE_L),  # shank+foot
    Segment("shank_foot_r", 0.0570, 0.45, KNEE_R, ANKLE_R),
)


# Joints at which we compute external moments (proximal joints of
# load-bearing segment chains).
TARGET_JOINTS = {
    "L5S1":     PELVIS,
    "shoulder_l": SHOULDER_L,
    "shoulder_r": SHOULDER_R,
    "elbow_l":  ELBOW_L,
    "elbow_r":  ELBOW_R,
    "hip_l":    HIP_L,
    "hip_r":    HIP_R,
    "knee_l":   KNEE_L,
    "knee_r":   KNEE_R,
}


# Which segments load each target joint (i.e. the distal kinematic
# chain whose weight the joint must support against gravity).
JOINT_DISTAL_SEGMENTS: dict[str, tuple[str, ...]] = {
    # spine base supports head + trunk + both arms + both hands
    "L5S1": ("head_neck", "trunk",
             "upper_arm_l", "upper_arm_r",
             "forearm_l", "forearm_r", "hand_l", "hand_r"),
    # shoulder supports one upper arm + forearm + hand
    "shoulder_l": ("upper_arm_l", "forearm_l", "hand_l"),
    "shoulder_r": ("upper_arm_r", "forearm_r", "hand_r"),
    "elbow_l":   ("forearm_l", "hand_l"),
    "elbow_r":   ("forearm_r", "hand_r"),
    "hip_l":     ("thigh_l", "shank_foot_l"),
    "hip_r":     ("thigh_r", "shank_foot_r"),
    "knee_l":    ("shank_foot_l",),
    "knee_r":    ("shank_foot_r",),
}


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

@dataclass
class JointMoment:
    joint: str
    joint_index: int
    torque_Nm: float                     # magnitude of net gravity moment
    torque_normalized: float             # τ / (M · g · H_ref)  (dimensionless)
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))  # unit vector
    load_kg: float = 0.0                 # gravitational load borne (kg-equiv)
    moment_arm_m: float = 0.0            # average horizontal moment arm
    contributing_segments: list[str] = field(default_factory=list)


@dataclass
class BiomechResult:
    body_mass_kg: float
    length_scale_m: float
    stature_m: float
    moments: dict[str, JointMoment]
    notes: list[str] = field(default_factory=list)

    def torque_dict(self) -> dict[str, float]:
        return {k: v.torque_Nm for k, v in self.moments.items()}

    def normalized_dict(self) -> dict[str, float]:
        return {k: v.torque_normalized for k, v in self.moments.items()}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _segment_com_position(seg: Segment, kp: np.ndarray) -> np.ndarray:
    """CoM position of `seg` in world coords given the major-25 pose."""
    p = kp[seg.proximal]
    d = kp[seg.distal]
    return p + seg.com_fraction * (d - p)


# Segment lengths as fractions of stature (Drillis and Contini), major-25 indices.
_STATURE_SEGMENTS = ((5, 7, 0.186), (6, 8, 0.186), (7, 9, 0.146), (8, 10, 0.146),
                     (11, 13, 0.245), (12, 14, 0.245), (13, 15, 0.246), (14, 16, 0.246),
                     (23, 24, 0.288))


def _estimate_stature(kp: np.ndarray) -> float:
    """Stature in pose units from image-plane segment lengths.

    Each limb and trunk segment gives a stature estimate (length / its share of
    stature). Foreshortening only shortens a segment, so an upper quantile is
    used; unlike the head-to-ankle span, the estimate does not shrink when the
    worker bends. ponytail: assumes a few limb segments lie near the image
    plane; a fully foreshortened pose still underestimates stature."""
    kp = np.asarray(kp, dtype=float)
    est = [np.hypot(*(kp[a, :2] - kp[b, :2])) / f for a, b, f in _STATURE_SEGMENTS]
    return float(np.percentile(est, 75))


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_static_moments(
    pose: Major25Pose,
    *,
    body_mass_kg: float = 70.0,
    length_scale_m: float = 1.0,
    gravity: float = 9.81,
    world_up: np.ndarray | None = None,
    external_loads: dict[str, tuple[float, np.ndarray]] | None = None,
) -> BiomechResult:
    """Compute static gravity moments at the major target joints.

    Args:
        pose: Major-25 keypoint pose.
        body_mass_kg: worker mass (kg).
        length_scale_m: multiplier converting pose-distance units to metres.
        gravity: m/s²; defaults to 9.81.
        world_up: unit vector pointing "up" in pose coordinates. The
            default `[0, -1, 0]` matches image-down (+Y) conventions.
        external_loads: optional `{joint_name: (mass_kg, application_point)}`
            entries that add hand-held / external loads.

    Returns:
        `BiomechResult` with per-joint torques (N·m and dimensionless).
    """
    if world_up is None:
        world_up = np.array([0.0, -1.0, 0.0])
    world_up = world_up / max(np.linalg.norm(world_up), 1e-9)
    gravity_vec = -world_up * gravity  # force-per-kg

    kp = pose.keypoints
    stature_pose = _estimate_stature(kp)
    stature_m = max(stature_pose * length_scale_m, 1e-6)

    # Pre-compute CoM positions for every segment (in pose units)
    com_pos: dict[str, np.ndarray] = {}
    for seg in SEGMENTS:
        com_pos[seg.name] = _segment_com_position(seg, kp)

    moments: dict[str, JointMoment] = {}
    for joint_name, joint_idx in TARGET_JOINTS.items():
        joint_pos = kp[joint_idx]
        net_moment = np.zeros(3, dtype=np.float64)
        total_force_mag = 0.0
        contrib_segs: list[str] = []
        load_kg_sum = 0.0
        weighted_arm = 0.0
        weight_sum = 0.0

        for seg_name in JOINT_DISTAL_SEGMENTS[joint_name]:
            seg = next(s for s in SEGMENTS if s.name == seg_name)
            mass = seg.mass_fraction * body_mass_kg
            r_pose = com_pos[seg_name] - joint_pos              # pose units
            r_m = r_pose * length_scale_m                       # metres
            force_N = mass * gravity_vec                        # Newtons (per kg)
            mom = np.cross(r_m, force_N)                        # N·m
            net_moment += mom
            total_force_mag += abs(mass * gravity)
            contrib_segs.append(seg_name)
            load_kg_sum += mass
            # horizontal moment arm: distance perpendicular to gravity
            arm = float(np.linalg.norm(r_m - np.dot(r_m, world_up) * world_up))
            weighted_arm += mass * arm
            weight_sum += mass

        # external load contribution
        if external_loads and joint_name in external_loads:
            ext_mass, ext_point_pose = external_loads[joint_name]
            r_m = (ext_point_pose - joint_pos) * length_scale_m
            mom = np.cross(r_m, -world_up * ext_mass * gravity)
            net_moment += mom
            contrib_segs.append("external")
            load_kg_sum += ext_mass

        tau = float(np.linalg.norm(net_moment))
        direction = (net_moment / tau) if tau > 1e-9 else np.zeros(3)
        norm = tau / (body_mass_kg * gravity * stature_m + 1e-9)

        moments[joint_name] = JointMoment(
            joint=joint_name,
            joint_index=joint_idx,
            torque_Nm=tau,
            torque_normalized=float(norm),
            direction=direction,
            load_kg=load_kg_sum,
            moment_arm_m=float(weighted_arm / max(weight_sum, 1e-9)),
            contributing_segments=contrib_segs,
        )

    return BiomechResult(
        body_mass_kg=body_mass_kg,
        length_scale_m=length_scale_m,
        stature_m=stature_m,
        moments=moments,
        notes=[f"gravity={gravity} m/s²", f"stature_proxy={stature_m:.3f} m"],
    )


# ----------------------------------------------------------------------
# Cohort utilities
# ----------------------------------------------------------------------

def aggregate_torques(results: Sequence[BiomechResult]) -> dict[str, dict[str, float]]:
    """Compute mean / median / std of each joint torque across a cohort."""
    joints = sorted(results[0].moments.keys())
    out = {}
    for j in joints:
        vals = np.array([r.moments[j].torque_Nm for r in results])
        out[j] = {
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
        }
    return out


def headline_torques(result: BiomechResult) -> dict[str, float]:
    """Compact "five-number" biomech summary for a row in the batch CSV."""
    return {
        "biomech_L5S1_Nm": result.moments["L5S1"].torque_Nm,
        "biomech_L5S1_norm": result.moments["L5S1"].torque_normalized,
        "biomech_shoulder_max_Nm": max(
            result.moments["shoulder_l"].torque_Nm,
            result.moments["shoulder_r"].torque_Nm,
        ),
        "biomech_knee_max_Nm": max(
            result.moments["knee_l"].torque_Nm,
            result.moments["knee_r"].torque_Nm,
        ),
        "biomech_total_load_kg": sum(
            m.load_kg for m in result.moments.values()
        ),
    }
