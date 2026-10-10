"""Model-schema definitions for the 9 keypoint conventions compared in
the BIOERA paper.

For each convention we provide:

  * `name` — display name
  * `is_real` — True if the model is run as a true ONNX/PyTorch
                inference; False if it is rendered as a *schematic*
                overlay derived from RTMW3D 133-pt detections via
                anatomical re-mapping.
  * `infer(image_bgr)` — returns (kpts_Mx2, scores_M, edges)
        for the model's body-only keypoint convention.

Real models: Body, BodyWithFeet, Wholebody3d (BIOERA).
Schematic models: H3.6M-17, H3.6M-32, SMPL-X body-22, BioPose-24,
HiPART (17→48), VEHS-66.

The schematic mapping uses the deterministic anatomical re-derivation
listed in `_remap_*` functions below. It is *not* a pretrained model
prediction; it shows what each convention's keypoint layout *would*
look like on the worker — useful for visual comparison of layout
density and coverage, but not for benchmarking accuracy.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from io import StringIO
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------
# rtmlib backbones
# ----------------------------------------------------------------------

def _silent(fn, *a, **kw):
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*a, **kw)


class _M:
    body = None
    bodyfeet = None
    wb3d = None


def _body():
    if _M.body is None:
        import rtmlib
        _M.body = _silent(rtmlib.Body, mode="balanced",
                          backend="onnxruntime", device="cuda")
    return _M.body


def _bodyfeet():
    if _M.bodyfeet is None:
        import rtmlib
        _M.bodyfeet = _silent(rtmlib.BodyWithFeet, mode="balanced",
                              backend="onnxruntime", device="cuda")
    return _M.bodyfeet


def _wb3d():
    if _M.wb3d is None:
        import rtmlib
        _M.wb3d = _silent(rtmlib.Wholebody3d, mode="balanced",
                          backend="onnxruntime", device="cuda")
    return _M.wb3d


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

# 2D body centroid of the analysed worker, set by the caller for each image so
# every model draws the same person. ponytail: module-level state; pass it
# through the infer functions if they are ever called concurrently.
TARGET_CENTROID: np.ndarray | None = None


def _primary(kpts: np.ndarray) -> int:
    if kpts.shape[0] == 0:
        return -1
    if kpts.shape[0] == 1 or TARGET_CENTROID is None:
        return 0
    cents = kpts[:, :17, :2].mean(axis=1)
    return int(np.argmin(np.linalg.norm(cents - TARGET_CENTROID, axis=1)))


# RTMW3D 133-pt indices we need:
# 0=nose, 1/2=L/R eye, 3/4=L/R ear, 5/6=L/R shoulder, 7/8=L/R elbow,
# 9/10=L/R wrist, 11/12=L/R hip, 13/14=L/R knee, 15/16=L/R ankle,
# 17/18/19=L big-toe/small-toe/heel, 20/21/22=R big-toe/small-toe/heel.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
L_BIG, L_SMA, L_HEE = 17, 18, 19
R_BIG, R_SMA, R_HEE = 20, 21, 22


def _derive_axes(k133: np.ndarray, s133: np.ndarray) -> dict:
    """Compute derived anatomical landmarks from RTMW3D 133-pt output."""
    pelvis = 0.5 * (k133[L_HIP] + k133[R_HIP])
    thorax = 0.5 * (k133[L_SHO] + k133[R_SHO])
    neck = 0.5 * (k133[NOSE] + thorax)            # rough neck base
    spine = 0.5 * (pelvis + thorax)
    head_top = k133[NOSE] + 1.5 * (k133[NOSE] - thorax) / max(
        np.linalg.norm(k133[NOSE] - thorax), 1e-6
    ) * 0.05 * np.linalg.norm(thorax - pelvis)
    l_clav = 0.5 * (thorax + k133[L_SHO])
    r_clav = 0.5 * (thorax + k133[R_SHO])
    return {
        "pelvis": pelvis, "thorax": thorax, "neck": neck, "spine": spine,
        "head_top": head_top, "l_clav": l_clav, "r_clav": r_clav,
        "pelvis_s": min(s133[L_HIP], s133[R_HIP]),
        "thorax_s": min(s133[L_SHO], s133[R_SHO]),
        "neck_s":   min(s133[NOSE], s133[L_SHO], s133[R_SHO]),
        "spine_s":  min(s133[L_HIP], s133[R_HIP], s133[L_SHO], s133[R_SHO]),
        "head_top_s": s133[NOSE] * 0.8,
        "l_clav_s": min(s133[L_SHO], s133[NOSE]) * 0.9,
        "r_clav_s": min(s133[R_SHO], s133[NOSE]) * 0.9,
    }


# ----------------------------------------------------------------------
# Real-model wrappers
# ----------------------------------------------------------------------

# COCO-17 body skeleton edges
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                   # face → ears
    (5, 6),                                            # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),                   # arms
    (5, 11), (6, 12), (11, 12),                        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
]
FOOT_EDGES_FROM_ANKLE = [(15, 17), (15, 18), (15, 19),
                          (16, 20), (16, 21), (16, 22)]


def infer_coco17(image_bgr):
    """RTMPose-COCO-17 via rtmlib.Body — real inference."""
    try:
        k, s = _silent(_body(), image_bgr)
    except Exception:
        return None
    if k.shape[0] == 0:
        return None
    i = _primary(k)
    return {"kpts": k[i][:17].copy(), "scores": s[i][:17].copy(),
            "edges": COCO17_EDGES}


def infer_halpe26(image_bgr):
    """BodyWithFeet (≈ Halpe-style body+feet) via rtmlib — real."""
    try:
        k, s = _silent(_bodyfeet(), image_bgr)
    except Exception:
        return None
    if k.shape[0] == 0:
        return None
    i = _primary(k)
    n = min(s.shape[1], 26)
    return {"kpts": k[i][:n].copy(), "scores": s[i][:n].copy(),
            "edges": COCO17_EDGES + FOOT_EDGES_FROM_ANKLE}


def infer_bioera25(image_bgr):
    """Wholebody3d → BIOERA major-25 — real RTMW3D inference."""
    try:
        k3, s, _simcc, k133 = _silent(_wb3d(), image_bgr)
    except Exception:
        return None
    if k133.shape[0] == 0:
        return None
    i = _primary(k133)
    kk = k133[i]
    ss = s[i]
    der = _derive_axes(kk, ss)
    # 23 body+feet from RTMW3D + derived neck (23) + derived pelvis (24)
    out_k = np.vstack([kk[:23], der["neck"], der["pelvis"]])
    out_s = np.concatenate([ss[:23], [der["neck_s"]], [der["pelvis_s"]]])
    edges = (COCO17_EDGES + FOOT_EDGES_FROM_ANKLE
             + [(23, 24), (5, 23), (6, 23), (11, 24), (12, 24)])
    return {"kpts": out_k, "scores": out_s, "edges": edges}


# ----------------------------------------------------------------------
# Schematic wrappers — derive each convention from RTMW3D
# ----------------------------------------------------------------------

# Reusable: get RTMW3D primary detection
def _get_rtmw_primary(image_bgr):
    try:
        _k3, s, _simcc, k133 = _silent(_wb3d(), image_bgr)
    except Exception:
        return None, None
    if k133.shape[0] == 0:
        return None, None
    i = _primary(k133)
    return k133[i], s[i]


def schematic_h36m17(image_bgr):
    """Human3.6M-17 layout, schematic."""
    k, s = _get_rtmw_primary(image_bgr)
    if k is None:
        return None
    der = _derive_axes(k, s)
    pts = np.stack([
        der["pelvis"],            # 0
        k[R_HIP], k[R_KNE], k[R_ANK],
        k[L_HIP], k[L_KNE], k[L_ANK],
        der["spine"], der["thorax"],
        k[NOSE], der["head_top"],
        k[L_SHO], k[L_ELB], k[L_WRI],
        k[R_SHO], k[R_ELB], k[R_WRI],
    ])
    scs = np.array([
        der["pelvis_s"], s[R_HIP], s[R_KNE], s[R_ANK],
        s[L_HIP], s[L_KNE], s[L_ANK],
        der["spine_s"], der["thorax_s"], s[NOSE], der["head_top_s"],
        s[L_SHO], s[L_ELB], s[L_WRI],
        s[R_SHO], s[R_ELB], s[R_WRI],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ]
    return {"kpts": pts, "scores": scs, "edges": edges}


def schematic_h36m32(image_bgr):
    """Human3.6M-32 = H3.6M-17 + extras (spine subdivisions, hand
    centres, foot detail). Schematic."""
    res = schematic_h36m17(image_bgr)
    if res is None:
        return None
    k, s = _get_rtmw_primary(image_bgr)
    der = _derive_axes(k, s)
    # Extra 15 points: 2 neck, 2 hand, 6 foot (toe+small_toe+heel × 2),
    # 4 spine subdivisions, 1 head extra
    pelvis = der["pelvis"]; thorax = der["thorax"]
    l1 = 0.75 * pelvis + 0.25 * thorax
    l3 = 0.50 * pelvis + 0.50 * thorax
    l5 = 0.30 * pelvis + 0.70 * thorax
    extras = np.stack([
        der["neck"], der["head_top"],
        k[L_WRI], k[R_WRI],                            # hand centres ≈ wrist
        k[L_BIG], k[L_SMA], k[L_HEE],
        k[R_BIG], k[R_SMA], k[R_HEE],
        l1, l3, l5, der["spine"], 0.5 * (der["spine"] + der["thorax"]),
    ])
    extra_s = np.array([
        der["neck_s"], der["head_top_s"],
        s[L_WRI], s[R_WRI],
        s[L_BIG], s[L_SMA], s[L_HEE],
        s[R_BIG], s[R_SMA], s[R_HEE],
        der["spine_s"], der["spine_s"], der["spine_s"],
        der["spine_s"], der["spine_s"],
    ])
    all_k = np.vstack([res["kpts"], extras])
    all_s = np.concatenate([res["scores"], extra_s])
    # Connect extras minimally to the spine
    edges = list(res["edges"])
    spine_idx = 7
    for i, ofs in enumerate([26, 27, 28, 29, 30]):  # spine extras
        edges.append((spine_idx, ofs))
    edges += [(13, 19), (16, 20)]                   # wrist → hand
    edges += [(6, 21), (6, 22), (6, 23)]            # L ankle → toes/heel
    edges += [(3, 24), (3, 25), (3, 26)]            # R ankle → toes/heel
    return {"kpts": all_k, "scores": all_s, "edges": edges}


def schematic_smplx22(image_bgr):
    """SMPL-X body-22 — pelvis, hips, spine1/2/3, knees, ankles, feet,
    neck, clavicles, head, shoulders, elbows, wrists. Schematic."""
    k, s = _get_rtmw_primary(image_bgr)
    if k is None:
        return None
    der = _derive_axes(k, s)
    pelvis = der["pelvis"]; thorax = der["thorax"]
    sp1 = 0.66 * pelvis + 0.34 * thorax
    sp2 = 0.50 * pelvis + 0.50 * thorax
    sp3 = 0.34 * pelvis + 0.66 * thorax
    head = 0.5 * (k[NOSE] + der["thorax"])
    pts = np.stack([
        pelvis,                                   # 0
        k[L_HIP], k[R_HIP],
        sp1, k[L_KNE], k[R_KNE],
        sp2, k[L_ANK], k[R_ANK],
        sp3,
        k[L_BIG], k[R_BIG],                       # L/R foot
        der["neck"], der["l_clav"], der["r_clav"],
        head,
        k[L_SHO], k[R_SHO],
        k[L_ELB], k[R_ELB],
        k[L_WRI], k[R_WRI],
    ])
    scs = np.array([
        der["pelvis_s"], s[L_HIP], s[R_HIP],
        der["spine_s"], s[L_KNE], s[R_KNE],
        der["spine_s"], s[L_ANK], s[R_ANK],
        der["spine_s"],
        s[L_BIG], s[R_BIG],
        der["neck_s"], der["l_clav_s"], der["r_clav_s"],
        s[NOSE], s[L_SHO], s[R_SHO],
        s[L_ELB], s[R_ELB], s[L_WRI], s[R_WRI],
    ])
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 5),
        (3, 6), (4, 7), (5, 8),
        (6, 9),
        (7, 10), (8, 11),
        (9, 12), (12, 13), (12, 14),
        (12, 15),                                # neck-head
        (13, 16), (14, 17),
        (16, 18), (17, 19),
        (18, 20), (19, 21),
    ]
    return {"kpts": pts, "scores": scs, "edges": edges}


def schematic_biopose24(image_bgr):
    """BioPose-24 = SMPL-X body-22 + L/R hand. Schematic."""
    res = schematic_smplx22(image_bgr)
    if res is None:
        return None
    k, s = _get_rtmw_primary(image_bgr)
    # add L hand and R hand at wrist (no finger keypoints in BIOERA-25)
    l_hand = k[L_WRI] + np.array([8.0, 0.0])
    r_hand = k[R_WRI] - np.array([8.0, 0.0])
    pts = np.vstack([res["kpts"], l_hand, r_hand])
    scs = np.concatenate([res["scores"], [s[L_WRI] * 0.9, s[R_WRI] * 0.9]])
    edges = list(res["edges"]) + [(20, 22), (21, 23)]
    return {"kpts": pts, "scores": scs, "edges": edges}


def schematic_hipart49(image_bgr):
    """HiPART hierarchical 17→48 — schematic: COCO-17 + ~32 dense
    intermediate joints (limb midpoints, elbow/knee lat/med, etc.)."""
    k, s = _get_rtmw_primary(image_bgr)
    if k is None:
        return None
    der = _derive_axes(k, s)
    pts_base = k[:17]
    s_base = s[:17]
    # 32 intermediate points
    # arms: upper-arm-mid, forearm-mid, elbow-lat/med, wrist-rad/ulnar (× 2 sides) = 12
    # legs: thigh-mid, shank-mid, knee-lat/med, ankle-lat/med (× 2 sides) = 12
    # spine: t1, t6, t12 (3) + clavicles (2)  = 5
    # extras: head-back (1)  + foot toes (2) = 3
    def mid(a, b): return 0.5 * (a + b)
    def perp_offset(a, b, sign=1, mag=4.0):
        v = b - a
        n = np.linalg.norm(v) + 1e-6
        perp = np.array([-v[1], v[0]]) / n
        return mid(a, b) + sign * perp * mag

    extras = [
        # L arm
        mid(k[L_SHO], k[L_ELB]),                  # upper-arm-mid L
        mid(k[L_ELB], k[L_WRI]),                  # forearm-mid L
        perp_offset(k[L_SHO], k[L_ELB], +1, 4),   # elbow-lat L
        perp_offset(k[L_SHO], k[L_ELB], -1, 4),   # elbow-med L
        perp_offset(k[L_ELB], k[L_WRI], +1, 4),   # wrist-rad L
        perp_offset(k[L_ELB], k[L_WRI], -1, 4),   # wrist-uln L
        # R arm
        mid(k[R_SHO], k[R_ELB]),
        mid(k[R_ELB], k[R_WRI]),
        perp_offset(k[R_SHO], k[R_ELB], +1, 4),
        perp_offset(k[R_SHO], k[R_ELB], -1, 4),
        perp_offset(k[R_ELB], k[R_WRI], +1, 4),
        perp_offset(k[R_ELB], k[R_WRI], -1, 4),
        # L leg
        mid(k[L_HIP], k[L_KNE]),
        mid(k[L_KNE], k[L_ANK]),
        perp_offset(k[L_HIP], k[L_KNE], +1, 4),
        perp_offset(k[L_HIP], k[L_KNE], -1, 4),
        perp_offset(k[L_KNE], k[L_ANK], +1, 4),
        perp_offset(k[L_KNE], k[L_ANK], -1, 4),
        # R leg
        mid(k[R_HIP], k[R_KNE]),
        mid(k[R_KNE], k[R_ANK]),
        perp_offset(k[R_HIP], k[R_KNE], +1, 4),
        perp_offset(k[R_HIP], k[R_KNE], -1, 4),
        perp_offset(k[R_KNE], k[R_ANK], +1, 4),
        perp_offset(k[R_KNE], k[R_ANK], -1, 4),
        # Spine (3)
        0.34 * der["pelvis"] + 0.66 * der["thorax"],
        0.50 * der["pelvis"] + 0.50 * der["thorax"],
        0.66 * der["pelvis"] + 0.34 * der["thorax"],
        # Clavicles (2)
        der["l_clav"], der["r_clav"],
        # Head extra (1)
        der["head_top"],
        # Foot toes (2)
        k[L_BIG], k[R_BIG],
    ]
    extras = np.stack(extras)
    extra_s = np.full(len(extras), 0.7)              # heuristic
    pts = np.vstack([pts_base, extras])
    scs = np.concatenate([s_base, extra_s])
    edges = list(COCO17_EDGES) + [
        # Connect intermediate joints to make the dense skeleton visible
        (5, 17), (17, 7), (7, 18), (18, 9),         # L upper-arm + forearm
        (6, 23), (23, 8), (8, 24), (24, 10),
        (11, 29), (29, 13), (13, 30), (30, 15),
        (12, 35), (35, 14), (14, 36), (36, 16),
    ]
    return {"kpts": pts, "scores": scs, "edges": edges}


def schematic_vehs66(image_bgr):
    """VEHS-6.7M 66-keypoint schematic: dense thoracic/lumbar spine,
    dense shoulder girdle, bilateral lat/med landmarks at elbow/knee/
    wrist/ankle."""
    k, s = _get_rtmw_primary(image_bgr)
    if k is None:
        return None
    der = _derive_axes(k, s)
    pelvis = der["pelvis"]; thorax = der["thorax"]
    # head 4
    head = [k[NOSE], der["head_top"],
            0.5 * (k[NOSE] + der["head_top"]),
            0.5 * (k[L_EAR] + k[R_EAR])]
    # neck 3
    neck = [der["neck"], thorax,
            0.5 * (der["neck"] + thorax)]
    # shoulder girdle 6
    shoulder = [k[L_SHO], k[R_SHO], der["l_clav"], der["r_clav"],
                k[L_SHO] + (k[L_SHO] - thorax) * 0.15,
                k[R_SHO] + (k[R_SHO] - thorax) * 0.15]
    # spine 8 (t1, t6, t12, l1, l3, l5, sacrum, thorax-mid)
    spine = []
    for w in np.linspace(0.0, 1.0, 8):
        spine.append((1 - w) * pelvis + w * thorax)
    # pelvis + hips 3
    pel = [pelvis, k[L_HIP], k[R_HIP]]
    # arms 12 (bilateral × 6)
    def perp_offset(a, b, sign=1, mag=4.0):
        v = b - a; n = np.linalg.norm(v) + 1e-6
        perp = np.array([-v[1], v[0]]) / n
        return 0.5 * (a + b) + sign * perp * mag

    arms = []
    for sho, elb, wri in [(L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI)]:
        arms += [
            0.5 * (k[sho] + k[elb]),
            0.5 * (k[elb] + k[wri]),
            k[elb],
            perp_offset(k[sho], k[elb], +1, 5),
            perp_offset(k[sho], k[elb], -1, 5),
            0.66 * k[elb] + 0.34 * k[wri],
        ]
    # wrists 6 (bilateral × 3)
    wrists = []
    for wri, elb in [(L_WRI, L_ELB), (R_WRI, R_ELB)]:
        wrists += [k[wri],
                   perp_offset(k[elb], k[wri], +1, 4),
                   perp_offset(k[elb], k[wri], -1, 4)]
    # hand 2
    hands = [k[L_WRI] + np.array([6.0, 0.0]),
             k[R_WRI] - np.array([6.0, 0.0])]
    # legs 12 (bilateral × 6)
    legs = []
    for hip, kne, ank in [(L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)]:
        legs += [
            0.5 * (k[hip] + k[kne]),
            0.5 * (k[kne] + k[ank]),
            k[kne],
            perp_offset(k[hip], k[kne], +1, 5),
            perp_offset(k[hip], k[kne], -1, 5),
            0.66 * k[kne] + 0.34 * k[ank],
        ]
    # ankles 6 (bilateral × 3)
    ankles = []
    for ank, kne in [(L_ANK, L_KNE), (R_ANK, R_KNE)]:
        ankles += [k[ank],
                   perp_offset(k[kne], k[ank], +1, 4),
                   perp_offset(k[kne], k[ank], -1, 4)]
    # feet 4
    feet = [k[L_BIG], k[R_BIG], k[L_HEE], k[R_HEE]]

    pts = np.stack(head + neck + shoulder + spine + pel + arms + wrists
                   + hands + legs + ankles + feet)
    scs = np.full(pts.shape[0], 0.7)
    # Stitch a stick skeleton
    n_head = 4; n_neck = 3; n_sho = 6; n_sp = 8; n_pel = 3
    o_sho = n_head + n_neck
    o_sp = o_sho + n_sho
    o_pel = o_sp + n_sp
    edges = []
    for i in range(n_sp - 1):
        edges.append((o_sp + i, o_sp + i + 1))
    edges.append((o_pel, o_sp))
    edges.append((o_sp + n_sp - 1, o_sho))
    return {"kpts": pts, "scores": scs, "edges": edges}


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    is_real: bool
    infer: Callable
    short: str


MODELS_9: list[ModelSpec] = [
    ModelSpec("COCO-17",       True,  infer_coco17,         "coco17"),
    ModelSpec("Halpe-26",      True,  infer_halpe26,        "halpe26"),
    ModelSpec("Human3.6M-17",  False, schematic_h36m17,     "h36m17"),
    ModelSpec("Human3.6M-32",  False, schematic_h36m32,     "h36m32"),
    ModelSpec("SMPL-X body-22", False, schematic_smplx22,   "smplx22"),
    ModelSpec("BioPose-24",    False, schematic_biopose24,  "biopose24"),
    ModelSpec("HiPART (17→48)", False, schematic_hipart49,  "hipart49"),
    ModelSpec("VEHS-6.7M (66)", False, schematic_vehs66,    "vehs66"),
    ModelSpec("BIOERA major-25", True, infer_bioera25,      "bioera25"),
]
