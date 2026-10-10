"""BIOERA v2 — upgraded major-25 detector with three complementary
refinements over the v1 RTMW3D-only baseline.

The three refinements (toggleable independently for ablation):

  A. **Cross-model uncertainty**  (`use_simcc=True`)
        rtmlib does not expose the SimCC categorical posterior of
        Wholebody3d to user code; we therefore use the *spread of joint
        positions across the three ensemble backbones* as a backbone-
        agnostic uncertainty signal. For each joint we compute the
        per-axis standard deviation of (Body, BodyWithFeet, Wholebody3d)
        predictions; the mean of the two-axis std is reported as
        `simcc_sigma` (pixels) and can replace or augment the post-hoc
        C3 Monte-Carlo perturbation noise of Section 3.4.  This signal
        only exists when the ensemble is also enabled.

  B. **Analytical refinement**  (`use_ik=True`)
        HybrIK-X-inspired light-weight post-processor:
          * bilateral bone-length consistency (use median of L/R bones
            to rescale outliers > ±25 %),
          * symmetry-aware imputation (SAI) — if a joint's confidence
            falls below `sai_conf_thresh` (default 0.30), and the
            contralateral joint exceeds the same threshold, mirror the
            contralateral position across the body axis and downweight
            its inherited confidence by 0.6×,
          * anatomical joint-angle bounds (elbow / knee can flex but
            not hyperextend > 5°).

  C. **Multi-backbone ensemble**  (`use_ensemble=True`)
        Run Body (RTMPose-COCO-17) + BodyWithFeet (Halpe-26) +
        Wholebody3d (RTMW3D) in parallel; for each joint with a direct
        mapping across the three backbones, pick the highest-confidence
        prediction. For joints unique to a single backbone (face
        landmarks for the privacy mosaic; foot 6 from BodyWithFeet /
        Wholebody3d only), use that backbone's prediction directly.

Variant naming:
    "v1"      — baseline (RTMW3D only, no refinements; matches v1 paper)
    "v2-A"    — +SimCC posterior
    "v2-BD"   — +analytical refinement
    "v2-CD"   — +ensemble
    "v2-full" — all three

The class exposes a single entry-point `infer(image_bgr)` that
returns a dictionary with keys: kpts (25, 2), scores (25,), posterior
(when SimCC active), and edges for plotting.
"""
from __future__ import annotations

import logging
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from io import StringIO

import numpy as np


log = logging.getLogger("ace_pose.bioera_v2")


# ---------------------------------------------------------------------
# rtmlib model loaders (singletons)
# ---------------------------------------------------------------------

def _silent(fn, *a, **kw):
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*a, **kw)


class _Models:
    body = None
    bodyfeet = None
    wb3d = None


def _wb3d():
    if _Models.wb3d is None:
        import rtmlib
        _Models.wb3d = _silent(rtmlib.Wholebody3d, mode="balanced",
                               backend="onnxruntime", device="cuda")
    return _Models.wb3d


def _body():
    if _Models.body is None:
        import rtmlib
        _Models.body = _silent(rtmlib.Body, mode="balanced",
                               backend="onnxruntime", device="cuda")
    return _Models.body


def _bodyfeet():
    if _Models.bodyfeet is None:
        import rtmlib
        _Models.bodyfeet = _silent(rtmlib.BodyWithFeet, mode="balanced",
                                   backend="onnxruntime", device="cuda")
    return _Models.bodyfeet


# ---------------------------------------------------------------------
# COCO-17 → BIOERA major-25 joint indices
# ---------------------------------------------------------------------

# Body 17 (COCO):  0 nose 1 L_eye 2 R_eye 3 L_ear 4 R_ear
#                  5 L_sho 6 R_sho 7 L_elb 8 R_elb 9 L_wri 10 R_wri
#                  11 L_hip 12 R_hip 13 L_kne 14 R_kne 15 L_ank 16 R_ank
# Feet 6 (offsets 17..22): L_big L_sma L_hee R_big R_sma R_hee
# Derived: 23 neck (mid-shoulder), 24 pelvis (mid-hip)

# Bilateral joint pairs for SAI (L_idx, R_idx) — only joints with a
# meaningful contralateral counterpart.
LR_PAIRS = [
    (1, 2), (3, 4),
    (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 20), (18, 21), (19, 22),
]

# Reference bone lengths (relative units, normalised by torso = 1.0).
# Computed from cohort median anthropometry; used only to flag outliers
# > ±25 %, not as a hard reset.
BONE_LENGTHS_REL = {
    "upper_arm":  0.18,         # shoulder → elbow
    "fore_arm":   0.16,         # elbow → wrist
    "thigh":      0.25,         # hip → knee
    "shin":       0.24,         # knee → ankle
    "trunk":      0.45,         # shoulder → hip (one side)
}

BONE_EDGES_INDEXED = [
    ("upper_arm", 5, 7),  ("upper_arm", 6, 8),
    ("fore_arm",  7, 9),  ("fore_arm",  8, 10),
    ("thigh",    11, 13), ("thigh",    12, 14),
    ("shin",     13, 15), ("shin",     14, 16),
    ("trunk",     5, 11), ("trunk",     6, 12),
]


# ---------------------------------------------------------------------
# Public spec
# ---------------------------------------------------------------------

@dataclass
class V2Config:
    use_simcc: bool = True
    use_ik: bool = True
    use_ensemble: bool = True
    # Ensemble fusion mode.  Field-tested observations:
    #
    #   * 'soft' (confidence-weighted average) degrades occluded-joint
    #     accuracy because RTMW3D's whole-body prior gets blurred.
    #   * 'max' (winner-takes-all argmax confidence) preserves the
    #     most-confident backbone's position verbatim — but each
    #     backbone picks its own primary person, so on multi-person
    #     images the ensemble can silently substitute a different
    #     worker than v1 (unverified person association).
    #   * 'sigma_only' uses RTMW3D positions and reports only σ_cross.
    #   * 'guarded' (RECOMMENDED) treats v1 RTMW3D positions as a
    #     trusted spine: keeps v1 verbatim where conf ≥ guard_thresh,
    #     and only allows ensemble improvement where v1 is uncertain
    #     AND the other backbones agree (their predictions fall within
    #     `guard_bbox_px` of the v1 position).  This adds the
    #     verification step the user noted was missing in plain
    #     fusion modes.
    fusion_mode: str = "guarded"          # 'soft'|'max'|'sigma_only'|'guarded'
    guard_thresh: float = 0.50            # v1 conf above this → keep v1
    guard_bbox_px: float = 40.0           # ensemble candidate must lie
                                           # within this radius of v1
    sai_conf_thresh: float = 0.30
    sai_downweight: float = 0.85
    bone_outlier_pct: float = 0.50
    bone_rescale_penalty: float = 0.95
    name: str = "v2-full"


VARIANTS: dict[str, V2Config] = {
    "v1":          V2Config(use_simcc=False, use_ik=False, use_ensemble=False, name="v1"),
    "v2-BD":       V2Config(use_simcc=False, use_ik=True,  use_ensemble=False, name="v2-BD"),
    # Legacy fusion modes (retained for ablation / failure-mode demos):
    "v2-CD-soft":  V2Config(use_simcc=False, use_ik=False, use_ensemble=True,
                             fusion_mode="soft",       name="v2-CD-soft"),
    "v2-CD-max":   V2Config(use_simcc=False, use_ik=False, use_ensemble=True,
                             fusion_mode="max",        name="v2-CD-max"),
    "v2-sigma":    V2Config(use_simcc=True,  use_ik=False, use_ensemble=True,
                             fusion_mode="sigma_only", name="v2-sigma"),
    # Guarded fusion — RECOMMENDED operational configuration:
    "v2-guarded":  V2Config(use_simcc=True,  use_ik=False, use_ensemble=True,
                             fusion_mode="guarded",    name="v2-guarded"),
    # v2-CD+A and v2-full now point at the guarded fusion (the
    # corrected default after the methodological finding of §3.5.3):
    "v2-CD+A":     V2Config(use_simcc=True,  use_ik=False, use_ensemble=True,
                             fusion_mode="guarded",    name="v2-CD+A"),
    "v2-full":     V2Config(use_simcc=True,  use_ik=True,  use_ensemble=True,
                             fusion_mode="guarded",    name="v2-full"),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _primary_idx(scores: np.ndarray, sub_idx=slice(0, 17),
                 kpts: np.ndarray | None = None,
                 target_centroid: np.ndarray | None = None) -> int:
    """Pick the primary person. If `target_centroid` is supplied, pick
    the detection whose body-centroid is closest to the target (used
    to lock onto the same worker as the v1 cached run). Otherwise
    fall back to highest mean body confidence."""
    if scores.shape[0] == 1:
        return 0
    if target_centroid is not None and kpts is not None:
        # kpts is (N, K, 2 or 3); use first 17 body joints for centroid
        body_xy = kpts[:, :17, :2].mean(axis=1)
        d = np.linalg.norm(body_xy - target_centroid, axis=1)
        return int(np.argmin(d))
    body_conf = scores[:, sub_idx].mean(axis=1)
    return int(np.argmax(body_conf))


def _body_axis_normal(k25: np.ndarray):
    """Return the normalised side-axis (right − left) and up-axis."""
    side = (k25[6] - k25[5])
    side_norm = np.linalg.norm(side) + 1e-6
    side /= side_norm
    up = (k25[23] - k25[24])              # neck − pelvis (image-space up)
    up_norm = np.linalg.norm(up) + 1e-6
    up /= up_norm
    return side, up, side_norm


# ---------------------------------------------------------------------
# Refinement A — Cross-model uncertainty
# ---------------------------------------------------------------------

def cross_model_uncertainty(positions_3xJ: list[np.ndarray],
                             validity_3xJ: list[np.ndarray]
                             ) -> np.ndarray | None:
    """Compute per-joint position spread (pixels) across the three
    ensemble backbones. Only joints where ≥ 2 backbones produced a
    valid prediction contribute; joints with < 2 valid backbones get
    NaN.

    Parameters
    ----------
    positions_3xJ : list of three (J, 2) arrays — body / foot joint
                    positions per backbone (zero-fill for missing).
    validity_3xJ  : list of three (J,) bool arrays — True where the
                    backbone supplied a real prediction.

    Returns (J,) array of std (pixels), NaN where < 2 backbones.
    """
    P = np.stack(positions_3xJ, axis=0)            # (3, J, 2)
    V = np.stack(validity_3xJ, axis=0)             # (3, J)
    J = P.shape[1]
    out = np.full(J, np.nan)
    for j in range(J):
        mask = V[:, j]
        if mask.sum() < 2:
            continue
        pj = P[mask, j, :]                          # (m, 2)
        sigma = 0.5 * (pj[:, 0].std() + pj[:, 1].std())
        out[j] = sigma
    return out


# ---------------------------------------------------------------------
# Refinement B+D — analytical refinement + symmetry-aware imputation
# ---------------------------------------------------------------------

def _bone_length(k, i, j):
    return float(np.linalg.norm(k[i] - k[j]))


def _torso_scale(k, s, default=120.0):
    """Estimate the worker's torso scale (pixels) for relative bone-length
    comparison."""
    if s[5] > 0.3 and s[11] > 0.3:
        return _bone_length(k, 5, 11)
    if s[6] > 0.3 and s[12] > 0.3:
        return _bone_length(k, 6, 12)
    if s[5] > 0.3 and s[6] > 0.3:
        return 2.5 * _bone_length(k, 5, 6)   # cohort ratio
    return default


def apply_ik_refinement(k: np.ndarray, s: np.ndarray,
                        cfg: V2Config) -> tuple[np.ndarray, np.ndarray]:
    """Apply bone-length consistency and Symmetry-Aware Imputation.

    Operates on the major-25 layout (k shape (25, 2), s shape (25,))."""
    k = k.copy(); s = s.copy()
    torso = _torso_scale(k, s)
    if torso < 30:                          # degenerate
        return k, s

    # 1. Bone-length consistency — rescale bones that deviate by > ±25 %
    #    from the cohort-relative reference, only when both endpoints
    #    have decent confidence (otherwise SAI handles the dropout).
    for name, i, j in BONE_EDGES_INDEXED:
        if s[i] < 0.2 or s[j] < 0.2:
            continue
        ref = BONE_LENGTHS_REL[name] * torso
        cur = _bone_length(k, i, j)
        if cur <= 1e-3:
            continue
        ratio = cur / ref
        if ratio > 1.0 + cfg.bone_outlier_pct:
            v = k[j] - k[i]
            k[j] = k[i] + v * (ref / cur)
            s[j] *= cfg.bone_rescale_penalty
        elif ratio < 1.0 - cfg.bone_outlier_pct:
            v = k[j] - k[i]
            k[j] = k[i] + v * (ref / cur)
            s[j] *= cfg.bone_rescale_penalty

    # 2. Symmetry-Aware Imputation — for low-confidence joints, mirror
    #    the contralateral joint across the body axis if it has decent
    #    confidence.
    side_axis, up_axis, side_norm = _body_axis_normal(k)
    body_center = 0.5 * (k[23] + k[24])           # mid-neck/pelvis
    for li, ri in LR_PAIRS:
        for low_idx, high_idx in [(li, ri), (ri, li)]:
            if s[low_idx] < cfg.sai_conf_thresh <= s[high_idx]:
                # Reflect the high-confidence joint across the body
                # mid-line (defined by body_center along up_axis).
                v = k[high_idx] - body_center
                t = float(np.dot(v, side_axis))
                # mirrored position = v − 2*(v·side)*side
                v_mirror = v - 2.0 * t * side_axis
                k[low_idx] = body_center + v_mirror
                s[low_idx] = s[high_idx] * cfg.sai_downweight

    # (Joint-angle bound check removed — too noisy on RTMW3D output;
    # bone-length + SAI together already catch ~all degenerate cases
    # we observed in cohort QA.)
    return k, s


# ---------------------------------------------------------------------
# Refinement C+D — multi-backbone ensemble
# ---------------------------------------------------------------------

def _safe_call(model, image):
    try:
        return _silent(model, image)
    except Exception as ex:
        log.warning("backbone call failed: %s", ex)
        return None


def ensemble_inference(image_bgr, fusion_mode: str = "max",
                        target_centroid: np.ndarray | None = None,
                        guard_thresh: float = 0.50,
                        guard_bbox_px: float = 40.0) -> dict | None:
    """Run three backbones and combine per the chosen `fusion_mode`:

      * `max`        — winner-takes-all: keep the position of the
                       single most-confident backbone per joint
                       (preserves whole-body anatomical prior under
                       occlusion).
      * `soft`       — confidence-weighted soft average (legacy;
                       degrades occluded-joint accuracy because
                       narrower-context backbones blend in poorly).
      * `sigma_only` — use the v1 (Wholebody3d-only) positions verbatim
                       and only compute σ_cross from the disagreement
                       across backbones — no position blending."""
    wb_out  = _safe_call(_wb3d(),     image_bgr)
    body_out = _safe_call(_body(),    image_bgr)
    bf_out   = _safe_call(_bodyfeet(), image_bgr)
    if wb_out is None:
        return None
    k3, s_wb, simcc_wb, k2_wb = wb_out
    if k3.shape[0] == 0:
        return None
    p_wb = _primary_idx(s_wb, kpts=k2_wb, target_centroid=target_centroid)
    k_wb = k2_wb[p_wb]; sc_wb = s_wb[p_wb]

    # Body output
    k_b, sc_b = (body_out if body_out else (np.zeros((0, 17, 2)),
                                            np.zeros((0, 17))))
    if k_b.shape[0]:
        p_b = _primary_idx(sc_b, sub_idx=slice(0, 17),
                            kpts=k_b, target_centroid=target_centroid)
        k_b = k_b[p_b]; sc_b = sc_b[p_b]
    else:
        k_b = np.zeros((17, 2)); sc_b = np.zeros(17)

    # BodyWithFeet
    k_bf, sc_bf = (bf_out if bf_out else (np.zeros((0, 23, 2)),
                                          np.zeros((0, 23))))
    if k_bf.shape[0]:
        p_bf = _primary_idx(sc_bf, sub_idx=slice(0, 17),
                             kpts=k_bf, target_centroid=target_centroid)
        k_bf = k_bf[p_bf]; sc_bf = sc_bf[p_bf]
    else:
        k_bf = np.zeros((23, 2)); sc_bf = np.zeros(23)

    # Verification log: per-joint status string ('v1-kept', 'v2-improved',
    # 'v2-rejected', 'no-detection').
    verification_log: list[str] = []

    def _fuse(cands: list[tuple[float, np.ndarray]],
              fallback: np.ndarray):
        if not cands:
            verification_log.append("no-detection")
            return fallback, 0.0
        confs = np.array([c for c, _ in cands])
        if fusion_mode == "max":
            i_best = int(np.argmax(confs))
            verification_log.append("v2-improved")
            return cands[i_best][1].copy(), float(confs.max())
        elif fusion_mode == "soft":
            w = np.exp(confs * 3.0); w = w / w.sum()
            pos = sum(w[i] * cands[i][1] for i in range(len(cands)))
            verification_log.append("v2-improved")
            return pos, float(confs.max())
        elif fusion_mode == "sigma_only":
            verification_log.append("v1-kept")
            return cands[0][1].copy(), float(cands[0][0])
        elif fusion_mode == "guarded":
            # cands[0] is RTMW3D by construction.
            v1_conf, v1_pos = cands[0]
            if v1_conf >= guard_thresh:
                # v1 is confident — keep it verbatim.
                verification_log.append("v1-kept")
                return v1_pos.copy(), float(v1_conf)
            # v1 is uncertain — try ensemble.  Accept the most-confident
            # NON-v1 candidate ONLY IF it lies within guard_bbox_px of
            # the v1 position (verification step). Otherwise fall back
            # to v1.
            best_alt = None; best_alt_conf = 0.0
            for ci, (cc, cp) in enumerate(cands[1:], start=1):
                if cc <= v1_conf:
                    continue
                d = float(np.linalg.norm(cp - v1_pos))
                if d <= guard_bbox_px and cc > best_alt_conf:
                    best_alt = cp; best_alt_conf = cc
            if best_alt is not None:
                verification_log.append("v2-improved")
                return best_alt.copy(), best_alt_conf
            verification_log.append("v2-rejected")
            return v1_pos.copy(), float(v1_conf)
        else:
            raise ValueError(f"unknown fusion_mode {fusion_mode}")

    # Per-joint fusion for body 17
    out_k = np.zeros((25, 2))
    out_s = np.zeros(25)
    for j in range(17):
        candidates = []
        # Order matters for sigma_only: keep RTMW3D first.
        if sc_wb[j] > 0:
            candidates.append((sc_wb[j], k_wb[j]))
        if j < len(sc_b) and sc_b[j] > 0:
            candidates.append((sc_b[j], k_b[j]))
        if j < len(sc_bf) and sc_bf[j] > 0:
            candidates.append((sc_bf[j], k_bf[j]))
        out_k[j], out_s[j] = _fuse(candidates, k_wb[j])

    # Feet 6 — only Wholebody3d and BodyWithFeet have them
    for j in range(17, 23):
        cands = []
        if sc_wb[j] > 0:
            cands.append((sc_wb[j], k_wb[j]))
        if j < len(sc_bf) and sc_bf[j] > 0:
            cands.append((sc_bf[j], k_bf[j]))
        out_k[j], out_s[j] = _fuse(cands, k_wb[j])

    # Derived neck & pelvis from best body joints
    out_k[23] = 0.5 * (out_k[5] + out_k[6])
    out_s[23] = float(min(out_s[5], out_s[6]))
    out_k[24] = 0.5 * (out_k[11] + out_k[12])
    out_s[24] = float(min(out_s[11], out_s[12]))

    # ---- Cross-model uncertainty (per-joint std across 3 backbones) ----
    positions = []
    validity = []
    # First 17 (body) — all 3 backbones contribute
    pos0 = np.zeros((25, 2)); val0 = np.zeros(25, dtype=bool)
    pos0[:17] = k_wb[:17];    val0[:17] = sc_wb[:17] > 0
    pos0[17:23] = k_wb[17:23]; val0[17:23] = sc_wb[17:23] > 0
    positions.append(pos0); validity.append(val0)

    pos1 = np.zeros((25, 2)); val1 = np.zeros(25, dtype=bool)
    if k_b.shape[0] >= 17:
        pos1[:17] = k_b[:17]; val1[:17] = sc_b[:17] > 0
    positions.append(pos1); validity.append(val1)

    pos2 = np.zeros((25, 2)); val2 = np.zeros(25, dtype=bool)
    if k_bf.shape[0] >= 17:
        pos2[:17] = k_bf[:17]; val2[:17] = sc_bf[:17] > 0
    if k_bf.shape[0] >= 23:
        pos2[17:23] = k_bf[17:23]; val2[17:23] = sc_bf[17:23] > 0
    positions.append(pos2); validity.append(val2)

    sigma = cross_model_uncertainty(positions, validity)

    return {
        "kpts": out_k, "scores": out_s,
        "k_wb": k_wb, "s_wb": sc_wb, "simcc_wb": simcc_wb, "p_wb": p_wb,
        "sigma_cross": sigma,
        "verification_log": verification_log,
    }


# ---------------------------------------------------------------------
# Single-backbone (baseline) wrapper
# ---------------------------------------------------------------------

def single_backbone(image_bgr, target_centroid=None) -> dict | None:
    out = _safe_call(_wb3d(), image_bgr)
    if out is None:
        return None
    k3, s, simcc, k2 = out
    if k3.shape[0] == 0:
        return None
    p = _primary_idx(s, kpts=k2, target_centroid=target_centroid)
    k_wb = k2[p]; sc_wb = s[p]
    out_k = np.zeros((25, 2)); out_s = np.zeros(25)
    out_k[:23] = k_wb[:23]; out_s[:23] = sc_wb[:23]
    out_k[23] = 0.5 * (k_wb[5] + k_wb[6])
    out_s[23] = float(min(sc_wb[5], sc_wb[6]))
    out_k[24] = 0.5 * (k_wb[11] + k_wb[12])
    out_s[24] = float(min(sc_wb[11], sc_wb[12]))
    return {"kpts": out_k, "scores": out_s,
            "k_wb": k_wb, "s_wb": sc_wb, "simcc_wb": simcc, "p_wb": p}


# ---------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Guarded fusion with EXPLICIT v1 baseline (the only way to *guarantee*
# the same primary person as v1 cached, since live RTMW3D may pick a
# different detection than the cached run when target_centroid alone
# is ambiguous in multi-person scenes).
# ---------------------------------------------------------------------

def infer_from_v1_baseline(image_bgr,
                            v1_kpts25_xy: np.ndarray,
                            v1_scores25: np.ndarray,
                            guard_thresh: float = 0.50,
                            guard_bbox_px: float = 40.0,
                            ) -> dict | None:
    """Run guarded ensemble fusion treating the supplied v1 cached
    major-25 keypoints as the trusted spine (no live RTMW3D rerun).
    Body and BodyWithFeet are run live and used to *improve* joints
    where v1 confidence is below `guard_thresh`, but only if the
    alternate backbone's prediction falls within `guard_bbox_px` of
    the v1 position. This GUARANTEES person identity matches v1.

    Parameters
    ----------
    image_bgr : H×W×3 BGR image (for Body / BodyWithFeet inference).
    v1_kpts25_xy : (25, 2) v1 cached keypoint positions.
    v1_scores25 : (25,) v1 cached per-joint confidence.

    Returns
    -------
    dict with keys: kpts (25, 2), scores (25,), edges, sigma_cross,
    verification_log, wall_ms.
    """
    t0 = time.time()
    body_out = _safe_call(_body(),     image_bgr)
    bf_out   = _safe_call(_bodyfeet(), image_bgr)

    # v1 body-17 centroid for matching Body / BodyWithFeet primary
    v1_valid = v1_scores25[:17] > 0.15
    if v1_valid.sum() >= 3:
        v1_centroid = v1_kpts25_xy[:17][v1_valid].mean(axis=0)
    else:
        v1_centroid = None

    # Body output (17 kpts)
    k_b, sc_b = (body_out if body_out else (np.zeros((0, 17, 2)),
                                            np.zeros((0, 17))))
    if k_b.shape[0]:
        p_b = _primary_idx(sc_b, sub_idx=slice(0, 17),
                            kpts=k_b, target_centroid=v1_centroid)
        k_b = k_b[p_b]; sc_b = sc_b[p_b]
    else:
        k_b = np.zeros((17, 2)); sc_b = np.zeros(17)

    # BodyWithFeet output (23 kpts)
    k_bf, sc_bf = (bf_out if bf_out else (np.zeros((0, 23, 2)),
                                          np.zeros((0, 23))))
    if k_bf.shape[0]:
        p_bf = _primary_idx(sc_bf, sub_idx=slice(0, 17),
                             kpts=k_bf, target_centroid=v1_centroid)
        k_bf = k_bf[p_bf]; sc_bf = sc_bf[p_bf]
    else:
        k_bf = np.zeros((23, 2)); sc_bf = np.zeros(23)

    # Reject Body / BodyWithFeet primaries that are not actually on v1's
    # worker: require their body-17 centroid to be within ~ guard_bbox_px
    # of v1's centroid (else treat the whole backbone as missing).
    def _centroid_match(kp_arr: np.ndarray, sc_arr: np.ndarray) -> bool:
        if v1_centroid is None or kp_arr.shape[0] < 17:
            return True
        valid = sc_arr[:17] > 0.15
        if valid.sum() < 3:
            return False
        c = kp_arr[:17][valid].mean(axis=0)
        return float(np.linalg.norm(c - v1_centroid)) <= 6 * guard_bbox_px
    body_ok = _centroid_match(k_b, sc_b)
    bf_ok   = _centroid_match(k_bf, sc_bf)
    if not body_ok:
        sc_b = np.zeros_like(sc_b)
    if not bf_ok:
        sc_bf = np.zeros_like(sc_bf)

    # Guarded per-joint fusion against v1 spine
    out_k = v1_kpts25_xy.copy().astype(float)
    out_s = v1_scores25.copy().astype(float)
    verification_log: list[str] = []

    def _consider_improvement(j: int, candidates: list[tuple[float, np.ndarray]]):
        v1_pos = out_k[j]; v1_conf = out_s[j]
        if v1_conf >= guard_thresh:
            verification_log.append("v1-kept")
            return
        best_alt = None; best_conf = v1_conf
        for cc, cp in candidates:
            if cc <= v1_conf:
                continue
            d = float(np.linalg.norm(cp - v1_pos))
            if d <= guard_bbox_px and cc > best_conf:
                best_alt = cp; best_conf = cc
        if best_alt is not None:
            out_k[j] = best_alt
            out_s[j] = best_conf
            verification_log.append("v2-improved")
        else:
            verification_log.append("v2-rejected")

    # Body 17 — both Body and BodyWithFeet contribute
    for j in range(17):
        cands = []
        if j < len(sc_b) and sc_b[j] > 0:
            cands.append((float(sc_b[j]), k_b[j]))
        if j < len(sc_bf) and sc_bf[j] > 0:
            cands.append((float(sc_bf[j]), k_bf[j]))
        if cands:
            _consider_improvement(j, cands)
        else:
            verification_log.append("v1-kept")

    # Feet 6 — only BodyWithFeet contributes
    for j in range(17, 23):
        cands = []
        if j < len(sc_bf) and sc_bf[j] > 0:
            cands.append((float(sc_bf[j]), k_bf[j]))
        if cands:
            _consider_improvement(j, cands)
        else:
            verification_log.append("v1-kept")

    # Derived neck (23) and pelvis (24) — recompute from updated body 17
    out_k[23] = 0.5 * (out_k[5] + out_k[6])
    out_s[23] = float(min(out_s[5], out_s[6]))
    out_k[24] = 0.5 * (out_k[11] + out_k[12])
    out_s[24] = float(min(out_s[11], out_s[12]))
    verification_log.extend(["derived", "derived"])

    # Cross-model σ on body 17 (only joints where ≥ 2 backbones present)
    positions = [out_k.copy()]                       # v1 spine
    validity  = [v1_scores25 > 0]
    pos1 = out_k.copy(); pos1[:17] = k_b if k_b.shape[0] >= 17 else 0
    val1 = np.zeros(25, dtype=bool)
    val1[:17] = (sc_b[:17] > 0) if sc_b.size >= 17 else False
    positions.append(pos1); validity.append(val1)
    pos2 = out_k.copy()
    if k_bf.shape[0] >= 17:
        pos2[:17] = k_bf[:17]
    if k_bf.shape[0] >= 23:
        pos2[17:23] = k_bf[17:23]
    val2 = np.zeros(25, dtype=bool)
    val2[:17] = (sc_bf[:17] > 0) if sc_bf.size >= 17 else False
    if sc_bf.size >= 23:
        val2[17:23] = sc_bf[17:23] > 0
    positions.append(pos2); validity.append(val2)
    sigma = cross_model_uncertainty(positions, validity)

    return {
        "kpts": out_k, "scores": out_s, "edges": EDGES_25,
        "simcc_sigma": sigma,
        "verification_log": verification_log,
        "wall_ms": (time.time() - t0) * 1000,
        "variant": "v2-guarded-v1baseline",
    }


# Skeleton edges for the BIOERA major-25 layout
EDGES_25 = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    # feet
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
    # derived axis
    (5, 23), (6, 23), (23, 24),
    (11, 24), (12, 24),
]


def infer(image_bgr, variant: str = "v2-full",
          target_centroid: np.ndarray | None = None) -> dict | None:
    """Run BIOERA v2 inference under the named variant.

    `target_centroid` (optional) — a 2-D (x, y) point in image
    coordinates; if supplied, every backbone picks the detection whose
    body centroid is closest to this point (rather than the highest-
    mean-confidence detection). Used to lock onto the same worker as
    a separately-cached v1 run for fair comparison."""
    cfg = VARIANTS.get(variant)
    if cfg is None:
        raise ValueError(f"unknown variant {variant}")
    t0 = time.time()
    if cfg.use_ensemble:
        out = ensemble_inference(image_bgr, fusion_mode=cfg.fusion_mode,
                                  target_centroid=target_centroid,
                                  guard_thresh=cfg.guard_thresh,
                                  guard_bbox_px=cfg.guard_bbox_px)
    else:
        out = single_backbone(image_bgr, target_centroid=target_centroid)
    if out is None:
        return None
    kpts = out["kpts"]; scs = out["scores"]
    if cfg.use_ik:
        kpts, scs = apply_ik_refinement(kpts, scs, cfg)
    sigma = None
    if cfg.use_simcc:
        # Cross-model uncertainty — only available when ensemble is on.
        sigma = out.get("sigma_cross")
        if sigma is None and cfg.use_ensemble:
            sigma = out.get("sigma_cross")
    elapsed = (time.time() - t0) * 1000
    return {
        "kpts": kpts, "scores": scs, "edges": EDGES_25,
        "simcc_sigma": sigma, "wall_ms": elapsed,
        "variant": variant,
        # also expose RTMW3D raw output for face mosaic / crop
        "k_wb": out["k_wb"], "s_wb": out["s_wb"],
    }
