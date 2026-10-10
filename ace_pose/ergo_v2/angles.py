"""Joint-angle computation from a major-25 pose.

All formulas derived from the documented standards in
`ace_pose/standards/{owas,rula,reba}.md §9` (or §4 for OWAS).

Conventions
-----------
Angles are measured in the image plane (+X right, +Y down), which is
treated as the sagittal plane. The monocular depth estimate is root-relative,
in different units from the image coordinates, and too noisy to orient the
body (Methods 3.3), so the depth coordinate is ignored. "Anterior" is the
image direction the worker faces, voted from the head (nose vs ears) and the
feet (toes vs ankles). Lateral bending, twist, abduction, and the RULA
midline / out-to-side flags are not observable in this plane and are
reported as 0 / False.

Output angles are in **degrees**. Flexion is positive (forward), extension
negative; lateral bending and twist are unsigned magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ace_pose.major_joints import (
    Major25Pose,
    NOSE, EYE_L, EYE_R, EAR_L, EAR_R,
    SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R,
    HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R,
    BIG_TOE_L, BIG_TOE_R, NECK, PELVIS,
)


# ----------------------------------------------------------------------
# Vector utilities
# ----------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle ∈ [0, 180] between two vectors."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _signed_angle_deg(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle ∈ [-180, 180] from u to v, around `axis`.

    Sign convention follows the right-hand rule with `axis` pointing to
    the viewer (positive sign for counter-clockwise rotation when looking
    along `-axis`).
    """
    u, v, axis = _norm(u), _norm(v), _norm(axis)
    if np.linalg.norm(u) < 1e-9 or np.linalg.norm(v) < 1e-9:
        return 0.0
    cross = np.cross(u, v)
    sin_a = float(np.dot(cross, axis))
    cos_a = float(np.dot(u, v))
    return float(np.degrees(np.arctan2(sin_a, cos_a)))


def _project_perpendicular(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of `v` perpendicular to `axis` (i.e., projected onto the
    plane normal to `axis`)."""
    axis = _norm(axis)
    return v - np.dot(v, axis) * axis


# ----------------------------------------------------------------------
# Body reference frame
# ----------------------------------------------------------------------

@dataclass
class BodyFrame:
    """Per-pose body-local coordinate frame.

    up:    pelvis → neck direction (the body's vertical, sign agnostic to
           camera orientation)
    side:  R_shoulder − L_shoulder (medio-lateral; positive = to subject's right)
    fwd:   cross(side, up) (anterior direction, away from subject's back)
    pelvis: origin point in input-frame coords
    """
    up: np.ndarray
    side: np.ndarray
    fwd: np.ndarray
    pelvis: np.ndarray
    planar: bool = False
    anterior: np.ndarray | None = None    # horizontal facing direction (planar frame)

    @property
    def valid(self) -> bool:
        return (
            np.linalg.norm(self.up) > 1e-6
            and np.linalg.norm(self.side) > 1e-6
            and np.linalg.norm(self.fwd) > 1e-6
        )


# Measurement tolerance for "upright" trunk and for neck extension: a few
# degrees of keypoint noise must not flip a neutral posture to a scored one.
UPRIGHT_TOL_DEG = 5.0
EXTENSION_LIMIT_DEG = 90.0   # upper-arm extension beyond this is an overhead raise


def facing_sign(pose: Major25Pose, min_conf: float = 0.3) -> float:
    """+1 if the worker faces image +X, -1 if image -X (votes: nose vs ears,
    big toes vs ankles); +1 when there is no evidence."""
    kp, c = pose.keypoints, pose.confidence
    vote = 0.0
    ears = [e for e in (EAR_L, EAR_R) if c[e] >= min_conf]
    if c[NOSE] >= min_conf and ears:
        vote += 2.0 * np.sign(kp[NOSE, 0] - np.mean([kp[e, 0] for e in ears]))
    for toe, ank in ((BIG_TOE_L, ANKLE_L), (BIG_TOE_R, ANKLE_R)):
        if c[toe] >= min_conf and c[ank] >= min_conf:
            vote += np.sign(kp[toe, 0] - kp[ank, 0])
    return -1.0 if vote < 0 else 1.0


def build_body_frame(pose: Major25Pose | np.ndarray) -> BodyFrame:
    """Image-plane (sagittal) body frame.

    `up`   = pelvis → neck in the image plane
    `side` = image normal, oriented so that `fwd` = cross(up, side)
    `fwd`  = in-plane perpendicular to `up`, pointing the way the worker faces
    """
    if isinstance(pose, Major25Pose):
        kp, sign = pose.keypoints, facing_sign(pose)
    else:
        kp, sign = np.asarray(pose), 1.0
    pelvis = np.array([kp[PELVIS, 0], kp[PELVIS, 1], 0.0])
    up = _norm(np.array([kp[NECK, 0] - kp[PELVIS, 0], kp[NECK, 1] - kp[PELVIS, 1], 0.0]))
    # fwd is `up` turned 90 degrees toward the facing side. Choosing its sign from
    # its own x component failed once the trunk bent past horizontal (2026-10-08).
    fwd = sign * np.array([-up[1], up[0], 0.0])
    side = np.array([0.0, 0.0, -sign])          # keeps fwd = cross(up, side)
    return BodyFrame(up=up, side=side, fwd=fwd, pelvis=pelvis, planar=True,
                     anterior=np.array([sign, 0.0, 0.0]))


# ----------------------------------------------------------------------
# Trunk
# ----------------------------------------------------------------------

def trunk_flexion_deg(pose: Major25Pose, frame: BodyFrame, world_up: np.ndarray | None = None) -> float:
    """Signed sagittal-plane trunk flexion (positive = anterior bend).

    Magnitude is the angle between the trunk vector (pelvis→neck) and the
    world up-axis. Sign is positive when the trunk leans anteriorly (in
    the +`frame.fwd` direction) and negative for posterior lean.
    """
    kp = pose.keypoints
    v_trunk = kp[NECK] - kp[PELVIS]
    if world_up is None:
        world_up = np.array([0, -1, 0])  # default: +Y is image-down
    world_up = _norm(world_up)
    # Project both onto the sagittal plane (perpendicular to frame.side)
    v_trunk_sag = _project_perpendicular(v_trunk, frame.side)
    world_up_sag = _project_perpendicular(world_up, frame.side)
    magnitude = _angle_deg(world_up_sag, v_trunk_sag)
    # lean toward the facing direction = flexion (fwd is perpendicular to the
    # trunk, so it cannot sign the trunk's own lean)
    anterior = frame.anterior if frame.anterior is not None else frame.fwd
    sign = 1.0 if np.dot(v_trunk_sag, anterior) >= 0 else -1.0
    return sign * magnitude


def trunk_twist_deg(pose: Major25Pose, frame: BodyFrame) -> float:
    """Unsigned angle (deg) between shoulder line and hip line in the
    transverse plane. Standard threshold for "twisted" is ≥10° (REBA),
    20° (OWAS)."""
    if frame.planar:
        return 0.0
    kp = pose.keypoints
    shoulder_line = kp[SHOULDER_R] - kp[SHOULDER_L]
    hip_line = kp[HIP_R] - kp[HIP_L]
    # Project both onto the transverse plane (perpendicular to frame.up)
    shoulder_t = _project_perpendicular(shoulder_line, frame.up)
    hip_t = _project_perpendicular(hip_line, frame.up)
    return abs(_signed_angle_deg(hip_t, shoulder_t, axis=frame.up))


def trunk_lateral_deg(pose: Major25Pose, frame: BodyFrame, world_up: np.ndarray | None = None) -> float:
    """Coronal-plane lateral bend angle of the trunk."""
    if frame.planar:
        return 0.0
    kp = pose.keypoints
    v_trunk = kp[NECK] - kp[PELVIS]
    if world_up is None:
        world_up = -frame.up if np.dot(frame.up, np.array([0, -1, 0])) > 0 else np.array([0, -1, 0])
    world_up = _norm(world_up)
    # project to coronal plane (perpendicular to frame.fwd)
    v_trunk_cor = _project_perpendicular(v_trunk, frame.fwd)
    world_up_cor = _project_perpendicular(world_up, frame.fwd)
    return abs(_signed_angle_deg(world_up_cor, v_trunk_cor, axis=frame.fwd))


# ----------------------------------------------------------------------
# Neck
# ----------------------------------------------------------------------

def neck_flexion_deg(pose: Major25Pose, frame: BodyFrame, world_up: np.ndarray | None = None) -> float:
    """Neck flexion relative to the trunk (RULA, REBA), positive = head bent
    anteriorly.

    Side views: head pitch from the ear -> eye line, which is about horizontal
    in neutral head posture (Frankfurt plane), minus the trunk flexion. Views
    without the eyes in front of the ears (from behind): angle between the
    trunk axis and the neck -> ear axis, which reads neutral as mild flexion.
    `world_up` is used for the trunk flexion only.
    """
    kp, c = pose.keypoints, pose.confidence
    eyes = [e for e in (EYE_L, EYE_R) if c[e] >= 0.3]
    ears = [e for e in (EAR_L, EAR_R) if c[e] >= 0.3]
    if frame.planar and eyes and ears and frame.anterior is not None:
        v = np.mean([kp[e] for e in eyes], axis=0) - np.mean([kp[e] for e in ears], axis=0)
        if v[0] * frame.anterior[0] > 0:          # eyes in front of the ears
            pitch = float(np.degrees(np.arctan2(v[1], abs(v[0]))))  # image y down: head pitched down
            return pitch - trunk_flexion_deg(pose, frame, world_up=world_up)
    ears = [e for e in (EAR_L, EAR_R) if c[e] >= 0.3]
    head_pt = np.mean([kp[e] for e in ears], axis=0) if ears else 0.5 * (kp[EYE_L] + kp[EYE_R])
    v_neck = _project_perpendicular(head_pt - kp[NECK], frame.side)
    v_trunk = _project_perpendicular(kp[NECK] - kp[PELVIS], frame.side)
    magnitude = _angle_deg(v_trunk, v_neck)
    across = v_neck - np.dot(v_neck, _norm(v_trunk)) * _norm(v_trunk)
    sign = 1.0 if np.dot(across, frame.fwd) > 0 else -1.0
    return sign * magnitude


def neck_twist_deg(pose: Major25Pose, frame: BodyFrame) -> float:
    """Yaw offset between head (ear line) and shoulders (transverse plane)."""
    if frame.planar:
        return 0.0
    kp = pose.keypoints
    ear_line = kp[EAR_R] - kp[EAR_L]
    shoulder_line = kp[SHOULDER_R] - kp[SHOULDER_L]
    ear_t = _project_perpendicular(ear_line, frame.up)
    shoulder_t = _project_perpendicular(shoulder_line, frame.up)
    return abs(_signed_angle_deg(shoulder_t, ear_t, axis=frame.up))


def neck_lateral_deg(pose: Major25Pose, frame: BodyFrame, world_up: np.ndarray | None = None) -> float:
    """Lateral bend of the neck (coronal plane)."""
    if frame.planar:
        return 0.0
    kp = pose.keypoints
    head_pt = 0.5 * (kp[EYE_L] + kp[EYE_R])
    v_neck = head_pt - kp[NECK]
    if world_up is None:
        world_up = -frame.up if np.dot(frame.up, np.array([0, -1, 0])) > 0 else np.array([0, -1, 0])
    world_up = _norm(world_up)
    v_neck_cor = _project_perpendicular(v_neck, frame.fwd)
    world_up_cor = _project_perpendicular(world_up, frame.fwd)
    return abs(_signed_angle_deg(world_up_cor, v_neck_cor, axis=frame.fwd))


# ----------------------------------------------------------------------
# Upper arm
# ----------------------------------------------------------------------

def upper_arm_flexion_deg(pose: Major25Pose, frame: BodyFrame, side: str = "right") -> float:
    """Sagittal upper-arm flexion. Positive when arm is anterior to trunk.

    Magnitude is the angle between the arm vector (shoulder→elbow) and
    the trunk-down reference. Sign is positive when the arm projects onto
    `+frame.fwd` (anterior to trunk), negative for extension behind.
    """
    kp = pose.keypoints
    sh = kp[SHOULDER_R] if side == "right" else kp[SHOULDER_L]
    el = kp[ELBOW_R] if side == "right" else kp[ELBOW_L]
    v_trunk_down = -frame.up
    v_arm = el - sh
    v_trunk_sag = _project_perpendicular(v_trunk_down, frame.side)
    v_arm_sag = _project_perpendicular(v_arm, frame.side)
    magnitude = _angle_deg(v_trunk_sag, v_arm_sag)
    sign = 1.0 if np.dot(v_arm_sag, frame.fwd) > 0 else -1.0
    if sign < 0 and magnitude > EXTENSION_LIMIT_DEG:
        # far beyond the shoulder's extension range: the arm is raised overhead,
        # past vertical, so measure it the anterior way round
        return 360.0 - magnitude
    return sign * magnitude


def upper_arm_abduction_deg(pose: Major25Pose, frame: BodyFrame, side: str = "right") -> float:
    """Coronal-plane abduction of the upper arm (unsigned)."""
    if frame.planar:
        return 0.0
    kp = pose.keypoints
    sh = kp[SHOULDER_R] if side == "right" else kp[SHOULDER_L]
    el = kp[ELBOW_R] if side == "right" else kp[ELBOW_L]
    v_trunk_down = -frame.up
    v_arm = el - sh
    v_trunk_cor = _project_perpendicular(v_trunk_down, frame.fwd)
    v_arm_cor = _project_perpendicular(v_arm, frame.fwd)
    return abs(_signed_angle_deg(v_trunk_cor, v_arm_cor, axis=frame.fwd))


# ----------------------------------------------------------------------
# Lower arm
# ----------------------------------------------------------------------

def elbow_flexion_deg(pose: Major25Pose, side: str = "right") -> float:
    """Elbow flexion (bend) in degrees: 0° when arm fully extended,
    increasing as the elbow bends. Defined as 180° − interior(thigh, shank).

    RULA/REBA score 1 is the "comfort zone" 60°–100° flexion (a typical
    relaxed elbow bend); above or below that range incurs +1.
    """
    kp = pose.keypoints
    if side == "right":
        u = kp[SHOULDER_R] - kp[ELBOW_R]
        v = kp[WRIST_R] - kp[ELBOW_R]
    else:
        u = kp[SHOULDER_L] - kp[ELBOW_L]
        v = kp[WRIST_L] - kp[ELBOW_L]
    interior = _angle_deg(u, v)
    return 180.0 - interior


def forearm_across_midline(pose: Major25Pose, frame: BodyFrame, side: str = "right") -> bool:
    """RULA / REBA: lower arm crosses midline of the body.

    Approximated by checking whether the wrist sits on the opposite side
    of the pelvis from the shoulder of the same arm.
    """
    if frame.planar:
        return False
    kp = pose.keypoints
    pelvis_to_shoulder_side = np.dot(
        (kp[SHOULDER_R] - kp[PELVIS]) if side == "right" else (kp[SHOULDER_L] - kp[PELVIS]),
        frame.side,
    )
    pelvis_to_wrist_side = np.dot(
        (kp[WRIST_R] - kp[PELVIS]) if side == "right" else (kp[WRIST_L] - kp[PELVIS]),
        frame.side,
    )
    return (pelvis_to_shoulder_side * pelvis_to_wrist_side) < 0


def forearm_out_to_side(pose: Major25Pose, frame: BodyFrame, side: str = "right",
                         stature_cm: float | None = None) -> bool:
    """RULA: lower arm works out to the side of the body.

    Uses the anthropometry module so the 15 % margin scales with the
    individual's stature when available (C2: anthropometric correction).
    """
    if frame.planar:
        return False
    from ace_pose.anthropometry import compute_anthropometry, is_arm_out_to_side
    anthro = compute_anthropometry(pose)
    return is_arm_out_to_side(pose, anthro, side=side, stature_cm=stature_cm)


# ----------------------------------------------------------------------
# Wrist (no finger keypoints → conservative defaults)
# ----------------------------------------------------------------------

WRIST_DEFAULT_RULA = 1     # unseen wrist taken as neutral, as the expert scores it (`standards/rula.md §10`)
WRIST_TWIST_DEFAULT_RULA = 1
WRIST_DEFAULT_REBA = 1     # per `standards/reba.md §12`


def wrist_flexion_deg(pose: Major25Pose, side: str = "right") -> float | None:
    """Without finger keypoints we cannot compute true wrist flexion;
    return None so the scoring layer uses its configurable default."""
    return None


# ----------------------------------------------------------------------
# Legs / knees
# ----------------------------------------------------------------------

def knee_flexion_deg(pose: Major25Pose, side: str = "right") -> float:
    """Knee flexion: 180° − interior angle between thigh and shank.

    Returns the FLEXION angle (0° for fully-extended knee, larger for
    more bent). Standard threshold for OWAS "straight" is ≤ 30° flexion
    (i.e., interior angle ≥ 150°).
    """
    kp = pose.keypoints
    if side == "right":
        thigh = kp[HIP_R] - kp[KNEE_R]
        shank = kp[ANKLE_R] - kp[KNEE_R]
    else:
        thigh = kp[HIP_L] - kp[KNEE_L]
        shank = kp[ANKLE_L] - kp[KNEE_L]
    interior = _angle_deg(thigh, shank)
    return 180.0 - interior


def legs_balanced(pose: Major25Pose, frame: BodyFrame) -> bool:
    """Weight distribution between the legs is not observable from one frame,
    so bilateral (balanced) support is assumed, like the OWAS walking flag;
    RULA/REBA callers can override via ``legs_supported``.

    The earlier ankle-symmetry heuristic labelled 84% of the cohort as
    one-legged, because the body frame's lateral axis is unstable in side
    views; it disagreed with the project ergonomist on 6 of 6 straight-legged
    standing poses it flagged (see standards/owas.md §2.3, 2026-10-05 note).
    """
    return True


def is_kneeling(pose: Major25Pose, frame: BodyFrame, threshold: float = 0.10) -> bool:
    """Either knee at floor level (close to a toe in length units).

    Uses the anthropometric helper so the threshold is robust to camera
    distance / worker stature (C2).
    """
    from ace_pose.anthropometry import compute_anthropometry, is_knee_at_floor
    anthro = compute_anthropometry(pose)
    return is_knee_at_floor(pose, anthro, "left") or is_knee_at_floor(pose, anthro, "right")


def is_sitting(pose: Major25Pose, frame: BodyFrame, hip_flex_threshold: float = 70.0) -> bool:
    """Both thighs near horizontal and both knees flexed.

    Thigh angle is taken from the image vertical, not from the trunk: measured
    against the trunk, a deep forward bend on straight legs read as sitting, and
    one bent knee was enough (2026-10-08).
    """
    kp = pose.keypoints
    down = np.array([0.0, 1.0, 0.0])                # image y grows downward
    for hip, knee in [(HIP_R, KNEE_R), (HIP_L, KNEE_L)]:
        if _angle_deg(down, kp[knee] - kp[hip]) < hip_flex_threshold:
            return False
    return knee_flexion_deg(pose, "left") >= 30 and knee_flexion_deg(pose, "right") >= 30


# ----------------------------------------------------------------------
# Arms above shoulder (for OWAS)
# ----------------------------------------------------------------------

def arms_above_shoulder(pose: Major25Pose, frame: BodyFrame) -> int:
    """Count of arms with wrist at or above shoulder level (0, 1, or 2)."""
    kp = pose.keypoints
    n = 0
    for sh, wr in [(SHOULDER_L, WRIST_L), (SHOULDER_R, WRIST_R)]:
        # +up direction is toward head; wrist above shoulder means
        # projection of (wrist - shoulder) onto frame.up is positive.
        if kp[wr, 1] <= kp[sh, 1]:          # image y grows downward
            n += 1
    return n
