"""3D pose from 2D keypoints under bone-length ratios and joint range-of-motion (ROM) limits.

A kinematic body model (pelvis, spine, head, arms, legs, feet) with fixed segment lengths
(fractions of stature) is fitted to the 2D keypoints of one image under weak perspective.
Every joint angle is bounded by its normal range of motion, so poses that a human body cannot
take (a knee or elbow bent backwards, a trunk twisted past its range) are not reachable.
A single image still leaves some bones ambiguous (toward or away from the camera); the fit
is started from several body headings and every solution whose 2D error is close to the best
is kept, so the spread of an angle across `FitResult.body_joints` is its uncertainty.

Coordinates of `forward()` and `FitResult.body_joints`: metres-free (stature = 1), y up,
z = the pelvis heading (forward), x = the subject's left. The camera is assumed level
(image vertical = world vertical), as in the image-plane scorer.

Segment lengths: fractions of stature after Drillis and Contini, as tabulated by Winter
(Biomechanics and Motor Control of Human Movement). ROM bounds: one normal range for every
worker, from ace_pose/standards/rom.md. ponytail: fixed population proportions, a per-worker length fit is the upgrade
if the length error turns out to dominate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ace_pose.major_joints import (
    ANKLE_L, ANKLE_R, BIG_TOE_L, BIG_TOE_R, EAR_L, EAR_R, ELBOW_L, ELBOW_R, EYE_L, EYE_R,
    HEEL_L, HEEL_R, HIP_L, HIP_R, KNEE_L, KNEE_R, NECK, NOSE, PELVIS, SHOULDER_L, SHOULDER_R,
    SMALL_TOE_L, SMALL_TOE_R, WRIST_L, WRIST_R,
)

SEGMENTS = {"trunk": 0.288, "shoulder_half": 0.1295, "hip_half": 0.0955, "upper_arm": 0.186,
            "forearm": 0.146, "thigh": 0.245, "shank": 0.246, "foot": 0.11, "heel": 0.04, "ankle_height": 0.039,
            "head": 0.118}
# Hand points in the hand frame (forearm axis -y, thumb +z, palm toward the body's midline):
# middle, index and little-finger knuckles (MCP). Hand length is 0.108 of stature (Winter); the
# knuckle positions below are model geometry (assumption), the same for ground truth and fit.
HAND = np.array([[0.0, -0.06, 0.0], [0.0, -0.058, 0.022], [0.0, -0.054, -0.022]])
# (name, lower, upper, neutral) in degrees; signs: flexion = forward, abduction = away from midline.
# Values and sources: ace_pose/standards/rom.md (VA 38 CFR 4.71/4.71a, one range for all sexes and
# ages). root_* are the pelvis orientation, not joints. Rows marked "assumption" have no source yet.
ROM = [("root_yaw", -360, 360, 0), ("root_pitch", -30, 120, 0), ("root_roll", -30, 30, 0),
       ("trunk_flex", -30, 90, 0), ("trunk_lat", -30, 30, 0), ("trunk_twist", -30, 30, 0),
       ("neck_flex", -45, 45, 0), ("neck_lat", -45, 45, 0), ("neck_rot", -80, 80, 0),
       ("shoulder_flex_r", -60, 180, 0), ("shoulder_flex_l", -60, 180, 0),      # extension: assumption
       ("shoulder_abd_r", -30, 180, 0), ("shoulder_abd_l", -30, 180, 0),        # adduction: assumption
       ("shoulder_rot_r", -90, 90, 0), ("shoulder_rot_l", -90, 90, 0),
       ("elbow_r", 0, 145, 0), ("elbow_l", 0, 145, 0),
       ("forearm_rot_r", -85, 80, 0), ("forearm_rot_l", -85, 80, 0),            # supination 85, pronation 80
       ("wrist_flex_r", -70, 80, 0), ("wrist_flex_l", -70, 80, 0),              # extension 70, flexion 80
       ("wrist_dev_r", -20, 45, 0), ("wrist_dev_l", -20, 45, 0),                # radial 20, ulnar 45
       ("hip_flex_r", -30, 125, 0), ("hip_flex_l", -30, 125, 0),                # extension: assumption
       ("hip_abd_r", -30, 45, 0), ("hip_abd_l", -30, 45, 0),                    # adduction: assumption
       ("hip_rot_r", -45, 45, 0), ("hip_rot_l", -45, 45, 0),                    # assumption
       ("knee_r", 0, 140, 0), ("knee_l", 0, 140, 0),
       ("ankle_r", -45, 20, 0), ("ankle_l", -45, 20, 0)]
PARAM_INDEX = {n: i for i, (n, *_) in enumerate(ROM)}
N_POSE = len(ROM)                      # followed by the camera: scale, image x, image y


def _rx(d):
    a = np.radians(d); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(d):
    a = np.radians(d); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(d):
    a = np.radians(d); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def neutral_params() -> np.ndarray:
    return np.array([n for *_, n in ROM], float)


def bounds():
    return np.array([lo for _, lo, _, _ in ROM], float), np.array([hi for _, _, hi, _ in ROM], float)


def forward(p: np.ndarray) -> np.ndarray:
    """Major-25 joints (25, 3) of the body model, heading removed (root yaw not applied)."""
    return _chain(p)[0]


def forward_hands(p: np.ndarray) -> np.ndarray:
    """(2, 3, 3): right then left hand; middle, index, little-finger knuckles."""
    return _chain(p)[1]


def _chain(p):
    g = dict(zip(PARAM_INDEX, p[:N_POSE]))
    hands = np.zeros((2, 3, 3))
    S, j = SEGMENTS, np.zeros((25, 3))
    down, up = np.array([0, -1.0, 0]), np.array([0, 1.0, 0])
    R_root = _rx(g["root_pitch"]) @ _rz(g["root_roll"])
    R_trunk = R_root @ _rx(g["trunk_flex"]) @ _rz(g["trunk_lat"]) @ _ry(g["trunk_twist"])
    j[NECK] = R_trunk @ (S["trunk"] * up)
    for sh, el, wr, x, s in ((SHOULDER_R, ELBOW_R, WRIST_R, -1, "r"), (SHOULDER_L, ELBOW_L, WRIST_L, 1, "l")):
        j[sh] = j[NECK] + R_trunk @ np.array([x * S["shoulder_half"], 0, 0])
        R_arm = R_trunk @ _rx(-g[f"shoulder_flex_{s}"]) @ _rz(x * g[f"shoulder_abd_{s}"]) @ _ry(g[f"shoulder_rot_{s}"])
        j[el] = j[sh] + R_arm @ (S["upper_arm"] * down)
        R_fore = R_arm @ _rx(-g[f"elbow_{s}"])
        j[wr] = j[el] + R_fore @ (S["forearm"] * down)
        # forearm rotation 0 = mid-position (thumb forward with the arm hanging), as VA Plate I
        R_hand = R_fore @ _ry(-x * g[f"forearm_rot_{s}"]) @ _rz(-x * g[f"wrist_flex_{s}"]) @ _rx(g[f"wrist_dev_{s}"])
        hands[0 if s == "r" else 1] = j[wr] + HAND @ R_hand.T
    R_head = R_trunk @ _rx(g["neck_flex"]) @ _rz(g["neck_lat"]) @ _ry(g["neck_rot"])
    head = j[NECK] + R_head @ (S["head"] * up)
    j[EAR_R], j[EAR_L] = head + R_head @ [-0.075, 0, -0.01], head + R_head @ [0.075, 0, -0.01]
    j[EYE_R], j[EYE_L] = head + R_head @ [-0.03, 0.01, 0.08], head + R_head @ [0.03, 0.01, 0.08]
    j[NOSE] = head + R_head @ [0, -0.02, 0.1]
    for hp, kn, an, toe, small, heel, x, s in ((HIP_R, KNEE_R, ANKLE_R, BIG_TOE_R, SMALL_TOE_R, HEEL_R, -1, "r"),
                                               (HIP_L, KNEE_L, ANKLE_L, BIG_TOE_L, SMALL_TOE_L, HEEL_L, 1, "l")):
        j[hp] = R_root @ np.array([x * S["hip_half"], 0, 0])
        R_leg = R_root @ _rx(-g[f"hip_flex_{s}"]) @ _rz(x * g[f"hip_abd_{s}"]) @ _ry(g[f"hip_rot_{s}"])
        j[kn] = j[hp] + R_leg @ (S["thigh"] * down)
        R_shank = R_leg @ _rx(g[f"knee_{s}"])
        j[an] = j[kn] + R_shank @ (S["shank"] * down)
        # +ankle = dorsiflexion (toes up); at 0 the sole is level, ankle_height above toe and heel
        R_foot = R_shank @ _rx(-g[f"ankle_{s}"])
        j[toe] = j[small] = j[an] + R_foot @ np.array([0, -S["ankle_height"], S["foot"]])
        j[heel] = j[an] + R_foot @ np.array([0, -S["ankle_height"], -S["heel"]])
    j[PELVIS] = 0.5 * (j[HIP_L] + j[HIP_R])
    return j, hands


def wrist_angles(elbow, wrist, middle, index, little, side: str):
    """(flexion, deviation) in degrees from the elbow, wrist and three knuckles: flexion positive
    toward the palm (negative = extension), deviation positive toward the little finger (ulnar).
    The same geometry is applied to ground truth and to fitted poses."""
    f = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    h = (middle - wrist) / np.linalg.norm(middle - wrist)
    u = little - index
    u = u - np.dot(u, h) * h
    u /= np.linalg.norm(u)
    n = np.cross(h, u) * (1.0 if side == "r" else -1.0)         # palm normal
    return (float(np.degrees(np.arctan2(-np.dot(f, n), np.hypot(np.dot(f, h), np.dot(f, u))))),
            float(np.degrees(np.arctan2(-np.dot(f, u), np.dot(f, h)))))


def camera_rotation(p: np.ndarray) -> np.ndarray:
    return _ry(p[PARAM_INDEX["root_yaw"]])


def project(j: np.ndarray, R: np.ndarray, scale: float = 1.0, t=(0.0, 0.0)) -> np.ndarray:
    """Weak perspective onto the image (x right, y down)."""
    w = j @ R.T
    return np.c_[scale * w[:, 0] + t[0], -scale * w[:, 1] + t[1]]


@dataclass
class FitResult:
    params: list            # one parameter vector per kept solution (pose then scale, tx, ty)
    rms: list               # 2D error of each, as a fraction of the 2D torso length
    body_joints: list       # (25, 3) joints of each, heading removed (y up, z forward)
    hand_joints: list = None  # (2, 3, 3) knuckles of each (right, left; middle, index, little)


def fit(kp2d: np.ndarray, conf: np.ndarray, min_conf: float = 0.3, tol: float = 0.03,
        reg: float = 0.01, headings=range(0, 360, 45), rom: bool = True,
        hands2d: np.ndarray | None = None, hands_conf: np.ndarray | None = None) -> FitResult:
    """Fit the body model to (25, 2) keypoints; keep every start whose error is within `tol`
    (fraction of the 2D torso length) of the best. rom=False opens every joint to +-180 degrees
    (the ablation without range-of-motion limits; segment lengths still fixed). hands2d (6, 2):
    right then left hand, knuckles in HAND order; without them the hand parameters stay near
    neutral (regularised) and the knuckles are not observed."""
    kp2d = np.asarray(kp2d, float)[:, :2]
    if hands2d is None:
        hands2d, hands_conf = np.zeros((6, 2)), np.zeros(6)
    hands2d = np.asarray(hands2d, float)[:, :2]
    wh = np.where(np.asarray(hands_conf) >= min_conf, np.sqrt(np.asarray(hands_conf)), 0.0)
    w = np.where(np.asarray(conf) >= min_conf, np.sqrt(np.asarray(conf)), 0.0)
    if np.sum(w > 0) < 6:
        return FitResult(params=[], rms=[], body_joints=[], hand_joints=[])
    # the 2D torso shrinks to nothing when the trunk points at the camera; fall back to body size
    torso = max(np.linalg.norm(kp2d[NECK] - kp2d[PELVIS]), 0.25 * np.ptp(kp2d[w > 0], axis=0).max(), 1e-6)
    lo, hi = bounds()
    span, neutral = hi - lo, neutral_params()
    if not rom:
        lo, hi = np.r_[lo[:1], np.full(N_POSE - 1, -180.0)], np.r_[hi[:1], np.full(N_POSE - 1, 180.0)]
    scale0 = torso / SEGMENTS["trunk"]
    lo_all = np.r_[lo, 0.2 * scale0, -np.inf, -np.inf]
    hi_all = np.r_[hi, 5.0 * scale0, np.inf, np.inf]

    def resid(q):
        j, hd = _chain(q)
        uv = project(np.r_[j, hd.reshape(6, 3)], camera_rotation(q), q[N_POSE], q[N_POSE + 1:])
        r = ((uv - np.r_[kp2d, hands2d]) * np.r_[w, wh][:, None]).ravel() / torso
        return np.r_[r, reg * (q[:N_POSE] - neutral) / span * np.sqrt(len(r))]

    sols = []
    for h in headings:
        q0 = np.r_[neutral, scale0, kp2d[PELVIS]]
        q0[PARAM_INDEX["root_yaw"]] = h
        r = least_squares(resid, q0, bounds=(lo_all, hi_all), x_scale=np.r_[span, scale0, torso, torso])
        e = resid(r.x)[:-N_POSE]
        sols.append((float(np.sqrt(np.sum(e ** 2) / max(np.sum(w > 0) + np.sum(wh > 0), 1))), r.x))
    best = min(s for s, _ in sols)
    keep = [(s, q) for s, q in sols if s <= best + tol]
    return FitResult(params=[q for _, q in keep], rms=[s for s, _ in keep],
                     body_joints=[forward(q) for _, q in keep], hand_joints=[forward_hands(q) for _, q in keep])
