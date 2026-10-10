"""Silhouette-based spinal curvature detection.

The pose-estimation literature (RTMW3D, Halpe-26, SMPL-X body, BioPose,
HiPART, …) cannot observe the spine directly — no public backbone emits
sacrum / L5 / L3 / T12 / T6 / T1 keypoints, and the Dense Spine
Estimation (DSE v1) module in `dense_spine.py` therefore inherits a
cohort-mean lumbar-lordosis template (White & Panjabi 1990) modulated by
the global trunk-flexion angle.  This works on the cohort average but
fails to distinguish worker-specific lumbar-vs-thoracic splits — the
exact clinical question that matters for disc-herniation risk.

This module implements **direct silhouette-based spinal curvature
measurement** when the worker's dorsal contour is visible:

  1. Segment the worker silhouette with YOLOv8-seg (ultralytics).
  2. Build a trunk-aligned orthonormal frame (trunk_axis = pelvis →
     neck; dorsal_dir = anti-anterior, perpendicular to trunk_axis).
  3. Walk *along the trunk axis* (not image Y) and at every sample
     measure the signed dorsal offset to the mask edge — this stays
     valid for bent-forward poses where image-Y is no longer the
     trunk direction.
  4. Gaussian-smooth the dorsal-offset signal, fit a B-spline to the
     reconstructed back curve, sample 9 vertebral levels along the
     spline at the de Leva 1996 cumulative trunk-length fractions, and
     read the local spline tangent as the per-vertebral flexion.

When the silhouette is occluded (cluttered factory background, frontal
view with no back contour visible) the module returns `None` and the
caller can fall back to the DSE v1 template.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np


log = logging.getLogger("ace_pose.spine_silhouette")


# Cumulative-trunk-length fractions for the 9 vertebral landmarks
# (same as DSE v1; sacrum = 0.0, mid-thorax = 1.0, occiput = 1.18).
SPINE_FRACTIONS = [
    ("sacrum",   0.00, "pelvis"),
    ("L5/S1",    0.04, "lumbar"),
    ("L3",       0.18, "lumbar"),
    ("L1",       0.32, "lumbar"),
    ("T12",      0.36, "thoraco-lumbar"),
    ("T6",       0.62, "thoracic"),
    ("T1",       0.92, "thoracic"),
    ("C7",       0.98, "cervical"),
    ("occiput",  1.05, "cranial"),
]


# Singleton YOLOv8-seg loader
class _Seg:
    model = None


def _load_segmenter():
    if _Seg.model is None:
        from ultralytics import YOLO
        log.info("Loading YOLOv8n-seg for silhouette extraction…")
        _Seg.model = YOLO("yolov8n-seg.pt")
    return _Seg.model


@dataclass
class SilhouetteSpine:
    levels: list[str]
    positions_xy: np.ndarray            # (L, 2) image-space (pixel)
    flex_deg: np.ndarray                # (L,) per-level flexion (degrees)
    region: list[str]
    spine_length_px: float
    fit_quality: float                  # 0..1 — how well silhouette was extractable
    back_curve_xy: np.ndarray | None = None  # (M, 2) dense back-surface samples (for diagnostic overlay)
    # ---- 4-keypoint trunk feedback (flexion + obliquity) ----
    trunk_flex_deg: float = 0.0          # trunk axis vs gravity-vertical
    obliquity_deg: float = 0.0           # shoulder girdle vs pelvis girdle (twist / side-bend)
    shoulder_xy: np.ndarray | None = None  # (2, 2) [L_sho, R_sho] px
    hip_xy: np.ndarray | None = None       # (2, 2) [L_hip, R_hip] px
    # Per-vertebral-level disc compression (N), aligned to `levels`,
    # computed from the 3-D pose; used to colour the v2 nodes by the
    # fixed NIOSH 1981 thresholds (worker-independent) while the line,
    # girdle and label stay a single fixed colour.
    node_compression_N: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


# =====================================================================
# 4-keypoint trunk-axis spine model (the operational estimator).
#
# Following the worker's *outer silhouette* conflates the spine with
# clothing bulk, carried loads and occluding limbs, so the
# contour-walk estimator above is error-prone in cluttered factory
# scenes.  The four trunk pose keypoints — both shoulders and both
# hips — are, by contrast, detected reliably by every backbone even
# when the silhouette is unusable.  This estimator therefore derives
# the spine *entirely* from those four points:
#
#   * pelvis  = midpoint(L_hip,  R_hip)        → sacrum / L5-S1 base
#   * thorax  = midpoint(L_sho,  R_sho)        → T1 / cervico-thoracic
#   * trunk axis + flexion angle θ from the pelvis→thorax chord vs
#     gravity (image-up)
#
# An earlier version superimposed a synthetic lordosis/kyphosis
# "S-curve" template.  That template needs the *sagittal facing
# direction* and *lumbar axial rotation*, neither of which is
# observable from four trunk keypoints — so the guessed S bulged the
# wrong way on oblique / rotated poses and looked jarringly
# unnatural.  We therefore report only what the keypoints actually
# measure: the trunk segment itself and its inclination.  The spine
# is the straight sacrum→T1 chord that follows the *measured* bend
# (it is not a fixed "1-자" — it tracks the worker's real trunk
# tilt), and the per-vertebral flexion is the honest, uniform trunk
# inclination rather than a fabricated per-level curve.
# =====================================================================

# (name, cumulative trunk fraction 0=sacrum→1=T1, region)
_KP_LEVELS = SPINE_FRACTIONS


def estimate_spine_from_keypoints(kp25_xy: np.ndarray,
                                    scores25: np.ndarray,
                                    conf_thresh: float = 0.20,
                                    ) -> SilhouetteSpine | None:
    """Trunk-axis spine from the four trunk keypoints (both shoulders
    + both hips).

    The spine is the sacrum→T1 chord that follows the *measured*
    trunk inclination; no synthetic sagittal curvature is added
    (orientation / axial rotation are not observable from four
    keypoints, so a fabricated S-curve would be misleading).

    Returns None only when one of the four trunk keypoints is missing
    or the trunk is degenerately short — the caller then falls back
    to the DSE v1 template."""
    L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
    quad = [L_SHO, R_SHO, L_HIP, R_HIP]
    if np.any(scores25[quad] < conf_thresh):
        return None

    l_sho = kp25_xy[L_SHO].astype(float)
    r_sho = kp25_xy[R_SHO].astype(float)
    l_hip = kp25_xy[L_HIP].astype(float)
    r_hip = kp25_xy[R_HIP].astype(float)
    thorax = 0.5 * (l_sho + r_sho)                    # T1 end
    pelvis = 0.5 * (l_hip + r_hip)                    # sacrum end
    trunk_vec = thorax - pelvis
    trunk_len = float(np.linalg.norm(trunk_vec))
    if trunk_len < 25.0:
        return None
    trunk_axis = trunk_vec / trunk_len

    # Trunk flexion θ: angle between the trunk axis and image-up
    # (0,-1).  Upright ≈ 0°, fully bent-over ≈ 90°.
    up = np.array([0.0, -1.0])
    cos_t = float(np.clip(np.dot(trunk_axis, up), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_t)))

    # Shoulder–pelvis obliquity (the trunk twist / side-bend signal
    # REBA & RULA flag and Marras 1995 identifies as a primary LBD
    # risk factor when combined with flexion).  The shoulder girdle
    # vector vs the pelvis girdle vector: when the worker twists or
    # side-bends the waist the two girdle lines rotate apart in the
    # image plane.  This is *measurable* from the four keypoints,
    # unlike the unobservable per-vertebra sagittal curve.
    sho_vec = r_sho - l_sho
    hip_vec = r_hip - l_hip
    if np.linalg.norm(sho_vec) > 1e-6 and np.linalg.norm(hip_vec) > 1e-6:
        cross = float(hip_vec[0] * sho_vec[1] - hip_vec[1] * sho_vec[0])
        dot = float(np.dot(hip_vec, sho_vec))
        obliquity_deg = abs(float(np.degrees(np.arctan2(cross, dot))))
        if obliquity_deg > 90.0:                 # fold onto [0, 90]
            obliquity_deg = 180.0 - obliquity_deg
    else:
        obliquity_deg = 0.0

    # Straight sacrum→T1 chord — follows the measured trunk tilt.
    t_dense = np.linspace(0.0, 1.0, 60)
    curve = pelvis[None, :] + t_dense[:, None] * trunk_vec[None, :]

    n_lv = len(_KP_LEVELS)
    positions = np.zeros((n_lv, 2))
    for i, (_nm, frac, _rg) in enumerate(_KP_LEVELS):
        tt = min(max(frac, 0.0), 1.0)
        positions[i] = pelvis + tt * trunk_vec
    # Per-level flexion = the honest, uniform trunk inclination.
    flex_deg = np.full(n_lv, theta_deg)

    fit_q = float(np.mean(scores25[quad]))
    return SilhouetteSpine(
        levels=[n for n, _, _ in _KP_LEVELS],
        positions_xy=positions,
        flex_deg=flex_deg,
        region=[r for _, _, r in _KP_LEVELS],
        spine_length_px=trunk_len,
        fit_quality=fit_q,
        back_curve_xy=curve,
        trunk_flex_deg=theta_deg,
        obliquity_deg=obliquity_deg,
        shoulder_xy=np.stack([l_sho, r_sho]),
        hip_xy=np.stack([l_hip, r_hip]),
        notes=[f"model=4kpt-trunk-axis",
               f"trunk_flex_deg={theta_deg:.1f}",
               f"obliquity_deg={obliquity_deg:.1f}",
               f"trunk_len_px={trunk_len:.1f}"],
    )


def _person_silhouette(model, image_bgr: np.ndarray, bbox=None):
    """Run YOLOv8-seg and return the largest person mask near `bbox`.

    If `bbox` (x0, y0, x1, y1) is given, prefer the detection whose
    centroid is closest to the bbox centre."""
    res = model.predict(image_bgr, classes=[0], verbose=False,
                         conf=0.30, iou=0.50)
    r = res[0]
    if r.masks is None or len(r.masks) == 0:
        return None, 0.0
    masks = r.masks.data.cpu().numpy()   # (N, H, W)
    confs = r.boxes.conf.cpu().numpy()
    h, w = image_bgr.shape[:2]
    if bbox is not None:
        cx_t = 0.5 * (bbox[0] + bbox[2]); cy_t = 0.5 * (bbox[1] + bbox[3])
        best = None; best_d = float("inf"); best_conf = 0.0
        for i in range(masks.shape[0]):
            m = masks[i]
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m > 0.5)
            if len(xs) < 100:
                continue
            cx = xs.mean(); cy = ys.mean()
            d = (cx - cx_t) ** 2 + (cy - cy_t) ** 2
            if d < best_d:
                best_d = d; best = m; best_conf = float(confs[i])
        if best is None:
            return None, 0.0
        return best.astype(bool), best_conf
    sizes = [m.sum() for m in masks]
    i = int(np.argmax(sizes))
    m = masks[i]
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool), float(confs[i])


def _trunk_aligned_back_curve(mask: np.ndarray,
                                pelvis: np.ndarray,
                                neck: np.ndarray,
                                anterior: np.ndarray,
                                shoulder_half_width: float,
                                n_samples: int = 60,
                                ) -> tuple[np.ndarray, np.ndarray] | None:
    """Walk *along the trunk axis* (not image Y) and return a smoothed
    dense back-curve plus the trunk-axis arc-length parameterisation.

    This is the key fix for bent-forward postures: the row-by-row Y
    walk used in the previous version cannot find the back surface of
    a worker bent at 80°, because the back is then on top of the
    image, not to one side.

    Returns
    -------
    (back_xy, s_arr): (M, 2) array of back-surface points in image
      coordinates and a parallel (M,) array of arc-length values along
      the trunk axis (0 = pelvis, |trunk| = neck), already smoothed
      with a Gaussian filter on the dorsal-offset signal.
    """
    trunk_vec = neck - pelvis
    trunk_len = float(np.linalg.norm(trunk_vec))
    if trunk_len < 30:
        return None
    trunk_axis = trunk_vec / trunk_len
    # Two image-plane perpendiculars to trunk_axis; pick the one anti-
    # parallel to the anterior direction (= dorsal).
    perp = np.array([-trunk_axis[1], trunk_axis[0]])
    anterior_u = anterior / max(np.linalg.norm(anterior), 1e-6)
    if float(np.dot(perp, anterior_u)) > 0:
        perp = -perp                                # flip to dorsal side

    # Maximum dorsal offset we accept.  On a deeply bent worker the
    # back surface arches a long way from the straight pelvis→neck
    # chord, so the bound is generous, but it is also capped at the
    # trunk length itself — the back surface physically cannot be
    # further from the chord than the trunk is long, and a larger
    # excursion means the walk escaped onto background clutter.
    max_off = min(4.0 * max(shoulder_half_width, 30.0),
                  1.0 * trunk_len)

    h, w = mask.shape
    # Sample strictly within the keypoint-defined trunk segment
    # (0 = pelvis, trunk_len = neck).  We do NOT extrapolate beyond the
    # neck: the occiput fraction (1.05) is clamped to 1.0 at sampling
    # time, and extrapolating a straight trunk axis into the head /
    # background produces the long phantom line the user flagged.
    s_arr = np.linspace(0.0, trunk_len, n_samples)
    dorsal_dist = np.full(n_samples, np.nan)
    back_xy = np.full((n_samples, 2), np.nan)

    for k, s in enumerate(s_arr):
        p_axis = pelvis + s * trunk_axis
        # Walk along dorsal direction from p_axis outward in unit-pixel
        # increments. Find the furthest still-in-mask point — that is
        # the dorsal surface at this trunk-axis position.
        max_d_in_mask = -1.0
        step = max(1.0, max_off / 200.0)
        d = 0.0
        while d <= max_off:
            px = int(round(p_axis[0] + d * perp[0]))
            py = int(round(p_axis[1] + d * perp[1]))
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                max_d_in_mask = d
            d += step
        # No anterior-side fallback: mixing dorsal and anterior hits
        # produces a contour that zig-zags across the body (the
        # F3-SUNTECH crouch artefact).  Rows with no dorsal pixel are
        # left NaN and linearly interpolated from their neighbours.
        if max_d_in_mask > 0:
            dorsal_dist[k] = max_d_in_mask
            back_xy[k] = (p_axis[0] + max_d_in_mask * perp[0],
                          p_axis[1] + max_d_in_mask * perp[1])

    valid = ~np.isnan(dorsal_dist)
    if valid.sum() < 0.6 * n_samples:
        return None

    # (Containment sanity gate is applied by the caller, which has the
    # worker's keypoint bounding box — a scale-free escape test that
    # does not collapse under trunk foreshortening on bent poses.)

    # Interpolate any missing samples linearly, then Gaussian-smooth
    # the dorsal-offset signal.
    idx = np.arange(n_samples)
    dorsal_dist = np.interp(idx, idx[valid], dorsal_dist[valid])

    # 3σ outlier rejection on first-difference: clip large jumps.
    diffs = np.diff(dorsal_dist)
    if diffs.size > 4:
        sd = np.std(diffs)
        clip = np.clip(diffs, -3.0 * sd, 3.0 * sd)
        dorsal_dist[1:] = dorsal_dist[0] + np.cumsum(clip)

    try:
        from scipy.ndimage import gaussian_filter1d
        sigma = max(2.0, n_samples / 25.0)
        dorsal_dist = gaussian_filter1d(dorsal_dist, sigma=sigma,
                                          mode="nearest")
    except Exception:
        # window-3 moving average as a tiny fallback
        kk = np.ones(3) / 3.0
        dorsal_dist = np.convolve(dorsal_dist, kk, mode="same")

    # Reconstruct the smoothed dense back curve in image space.
    p_axes = pelvis[None, :] + s_arr[:, None] * trunk_axis[None, :]
    back_xy = p_axes + dorsal_dist[:, None] * perp[None, :]
    return back_xy, s_arr


def estimate_spine_from_silhouette(image_bgr: np.ndarray,
                                     kp25_xy: np.ndarray,
                                     scores25: np.ndarray,
                                     bbox=None,
                                     gravity_deg_ref: float = 0.0,
                                     ) -> SilhouetteSpine | None:
    """Direct spinal-curvature estimation from worker silhouette.

    Returns None when segmentation fails or the back contour cannot be
    reliably extracted (caller should fall back to DSE v1 template)."""
    model = _load_segmenter()
    L_SHO, R_SHO, L_HIP, R_HIP, NOSE = 5, 6, 11, 12, 0
    pelvis = 0.5 * (kp25_xy[L_HIP] + kp25_xy[R_HIP])
    thorax = 0.5 * (kp25_xy[L_SHO] + kp25_xy[R_SHO])
    anterior = kp25_xy[NOSE] - thorax
    if np.linalg.norm(anterior) < 8:
        return None
    body_pts = kp25_xy[[L_SHO, R_SHO, L_HIP, R_HIP]]
    bx0 = float(min(body_pts[:, 0])); bx1 = float(max(body_pts[:, 0]))
    by0 = float(min(body_pts[:, 1])); by1 = float(max(body_pts[:, 1]))
    pad = 0.5 * (bx1 - bx0)
    target_bbox = (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
    if bbox is None:
        bbox = target_bbox

    mask, mask_conf = _person_silhouette(model, image_bgr, bbox=bbox)
    if mask is None or mask_conf < 0.30:
        return None

    shoulder_half_width = 0.5 * float(np.linalg.norm(
        kp25_xy[L_SHO] - kp25_xy[R_SHO]))
    if shoulder_half_width < 15:
        shoulder_half_width = 0.5 * float(np.linalg.norm(
            kp25_xy[L_HIP] - kp25_xy[R_HIP]))

    walk = _trunk_aligned_back_curve(mask, pelvis, thorax, anterior,
                                       shoulder_half_width)
    if walk is None:
        return None
    back_xy, s_arr = walk
    if back_xy.shape[0] < 8:
        return None

    # Containment sanity gate (scale-free).  The spine curve must stay
    # within the worker's own keypoint bounding box, generously
    # expanded.  When the silhouette walk escapes onto a neighbouring
    # worker or background clutter (the F3-SUNTECH crouch-against-a-
    # standing-person failure) the curve shoots outside this box; we
    # then reject and let the caller fall back to the v1 template.
    kp_conf = kp25_xy[scores25 > 0.30]
    if kp_conf.shape[0] >= 4:
        kx0, ky0 = kp_conf.min(axis=0)
        kx1, ky1 = kp_conf.max(axis=0)
        mx = 0.55 * (kx1 - kx0); my = 0.55 * (ky1 - ky0)
        inside = ((back_xy[:, 0] >= kx0 - mx) & (back_xy[:, 0] <= kx1 + mx)
                  & (back_xy[:, 1] >= ky0 - my) & (back_xy[:, 1] <= ky1 + my))
        if inside.mean() < 0.85:
            return None

    # Fit a B-spline to the smoothed back curve and sample 9 vertebral
    # levels at the de Leva cumulative-trunk-length fractions along the
    # spline's own arc length.
    try:
        from scipy.interpolate import splprep, splev
        # The dorsal-offset signal is already Gaussian-smoothed, so the
        # spline only needs a light smoothing factor; a large `s`
        # over-relaxes the endpoints and lets the curve overshoot past
        # the neck / sacrum.
        tck, _u = splprep([back_xy[:, 0], back_xy[:, 1]],
                          s=max(5.0, back_xy.shape[0] * 0.1), k=3)
        # Dense sampling along the spline for arc-length integration
        u_dense = np.linspace(0.0, 1.0, 400)
        xd, yd = splev(u_dense, tck)
        d = np.diff(np.stack([xd, yd], axis=-1), axis=0)
        seg = np.sqrt((d ** 2).sum(axis=-1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = float(cum[-1])
        # 9 vertebral positions + spline tangent at each
        positions = np.zeros((len(SPINE_FRACTIONS), 2))
        flex = np.zeros(len(SPINE_FRACTIONS))
        for j, (_name, frac, _r) in enumerate(SPINE_FRACTIONS):
            target = max(0.0, min(1.0, frac)) * total_len
            i = int(np.searchsorted(cum, target, side="right") - 1)
            i = max(0, min(i, u_dense.size - 2))
            positions[j] = (xd[i], yd[i])
            # spline tangent via numerical derivative at u_dense[i]
            du = 1.0 / 400.0
            t = np.array([(xd[min(i + 1, u_dense.size - 1)] - xd[max(i - 1, 0)]),
                          (yd[min(i + 1, u_dense.size - 1)] - yd[max(i - 1, 0)])])
            n = float(np.linalg.norm(t))
            if n < 1e-6:
                flex[j] = 0.0
                continue
            t = t / n
            # Flexion angle: angle between local tangent and the GLOBAL
            # gravity vector (image-down (0, +1) in image coords).
            ang = float(np.degrees(np.arctan2(abs(t[0]), abs(t[1]))))
            flex[j] = ang
        back_curve = np.stack([xd, yd], axis=-1)
    except Exception as e:
        log.warning("spline fit failed: %s — using polyline samples", e)
        # Polyline fallback
        d = np.diff(back_xy, axis=0)
        seg = np.sqrt((d ** 2).sum(axis=-1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = float(cum[-1])
        positions = np.zeros((len(SPINE_FRACTIONS), 2))
        flex = np.zeros(len(SPINE_FRACTIONS))
        for j, (_name, frac, _r) in enumerate(SPINE_FRACTIONS):
            target = max(0.0, min(1.0, frac)) * total_len
            i = int(np.searchsorted(cum, target, side="right") - 1)
            i = max(0, min(i, back_xy.shape[0] - 2))
            positions[j] = back_xy[i]
            t = back_xy[i + 1] - back_xy[i]
            n = float(np.linalg.norm(t))
            if n < 1e-6:
                flex[j] = 0.0; continue
            t = t / n
            flex[j] = float(np.degrees(np.arctan2(abs(t[0]), abs(t[1]))))
        back_curve = back_xy

    fit_quality = mask_conf * (1.0 if back_xy.shape[0] > 30 else
                                back_xy.shape[0] / 30.0)
    return SilhouetteSpine(
        levels=[name for name, _, _ in SPINE_FRACTIONS],
        positions_xy=positions,
        flex_deg=flex,
        region=[r for _, _, r in SPINE_FRACTIONS],
        spine_length_px=total_len,
        fit_quality=float(fit_quality),
        back_curve_xy=back_curve,
        notes=[f"mask_conf={mask_conf:.2f}",
               f"back_samples={back_xy.shape[0]}",
               f"trunk_len_px={total_len:.1f}"],
    )
