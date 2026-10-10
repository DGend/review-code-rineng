"""Shared helper to overlay the silhouette-based spinal curve (DSE v2,
`ace_pose.spine_silhouette`) on the manuscript figure panels.

The four figure scripts (pose_overview, pose_load_overview,
v2_ablation_overlay, v1_vs_v2_clinical) all follow the same
cv2-draw-then-crop pattern, so the spine curve is drawn on the
full-resolution BGR image *before* the crop — the crop math in each
caller is therefore unchanged.

YOLOv8n-seg is the expensive step. The silhouette is a property of the
*image / worker*, not of the keypoint-fusion variant, so the result is
cached on disk keyed by file stem and reused across every variant panel
of the same worker (the v1→v2 variants all track the same person by
construction — v2-guarded explicitly locks to the v1 baseline).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from ace_pose.spine_silhouette import (
    estimate_spine_from_keypoints, SilhouetteSpine,
)


log = logging.getLogger("ace_pose.spine_overlay")

# Single fixed colour for the whole DSE v2 overlay (trunk axis +
# girdles + markers + label). Severity is conveyed by the numeric
# `Flex / Obl` label, NOT by colour, so the overlay reads identically
# on every worker (the per-worker colour change was non-intuitive).
_HALO_BGR = (30, 30, 30)
_SPINE_BGR = (230, 200, 40)         # cyan — v2 trunk axis (fixed)
_GIRDLE_BGR = (230, 200, 40)        # cyan — shoulder & hip girdle lines


def kp25_px_from_133(k2: np.ndarray, sc: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Build the major-25 layout in PIXEL space from the RTMW3D 133-pt
    detection (the verified pixel path — `kpts_m25` in the npz is the
    3D-normalised variant and is unsuitable for the image-space
    silhouette walk)."""
    kp = np.zeros((25, 2), dtype=float)
    cf = np.zeros(25, dtype=float)
    kp[:23] = k2[:23, :2]
    cf[:23] = sc[:23]
    kp[23] = 0.5 * (k2[5, :2] + k2[6, :2])          # derived neck
    cf[23] = min(sc[5], sc[6])
    kp[24] = 0.5 * (k2[11, :2] + k2[12, :2])         # derived pelvis
    cf[24] = min(sc[11], sc[12])
    return kp, cf


def silhouette_spine_cached(image_bgr: np.ndarray,
                              kp25_px: np.ndarray,
                              scores25: np.ndarray,
                              cache_dir: Path | None = None,
                              stem: str = "",
                              ) -> SilhouetteSpine | None:
    """Return the anatomical spine for this worker from the four trunk
    keypoints (both shoulders + both hips).

    The estimator is now keypoint-driven (not silhouette-driven): it is
    cheap, deterministic and image-independent, so no on-disk cache is
    needed.  The signature is kept (incl. the unused `image_bgr`,
    `cache_dir`, `stem`) so the four figure callers do not change."""
    return estimate_spine_from_keypoints(kp25_px, scores25)


# Fixed NIOSH 1981 node bands (worker-independent — same compression
# always gives the same colour, so the overlay stays consistent).
_NODE_SAFE_BGR = _SPINE_BGR             # < AL  → same cyan as the line
_NODE_AL_BGR = (0, 165, 255)            # AL–MPL → orange ( > 3400 N )
_NODE_MPL_BGR = (38, 38, 220)           # ≥ MPL  → red    ( ≥ 6400 N )


def attach_node_load(spine: SilhouetteSpine | None,
                      kp25_m25_3d: np.ndarray,
                      conf_m25: np.ndarray,
                      body_mass_kg: float = 70.0,
                      ) -> SilhouetteSpine | None:
    """Compute the physically-scaled per-vertebral-level disc
    compression (N) from the **3-D** pose and attach it to
    `spine.node_compression_N` (aligned to `spine.levels`), so the v2
    nodes can be coloured by the fixed NIOSH 1981 thresholds.

    Mirrors the `run_disc_load` pipeline exactly: stature-based length
    scale (1.72 m / estimated stature) → `dense_spine` per-level
    moment → Chaffin single-equivalent-muscle compression at the 3-D
    trunk-flexion angle.  No-op when `spine` is None."""
    if spine is None:
        return spine
    from ace_pose.biomechanics import _estimate_stature
    from ace_pose.dense_spine import estimate_dense_spine
    from ace_pose.disc_load import (
        disc_compression_N, trunk_flexion_deg_3d,
    )
    st = float(_estimate_stature(kp25_m25_3d))
    ls = 1.72 / st if st > 1e-6 else 1.0
    ds = estimate_dense_spine(kp25_m25_3d, conf_m25,
                              body_mass_kg=body_mass_kg,
                              length_scale_m=ls)
    flex = trunk_flexion_deg_3d(kp25_m25_3d, conf_m25)
    comp = disc_compression_N(ds, body_mass_kg, flex)
    spine.node_compression_N = np.array(
        [float(comp.get(lv, 0.0)) for lv in spine.levels], dtype=float)
    return spine


def _node_bgr(comp_n: float):
    """Fixed NIOSH 1981 node colour: < AL cyan, AL–MPL orange,
    ≥ MPL red.  Same threshold on every level and every worker."""
    from ace_pose.disc_load import NIOSH_AL_N, NIOSH_MPL_N
    if comp_n >= NIOSH_MPL_N:
        return _NODE_MPL_BGR
    if comp_n >= NIOSH_AL_N:
        return _NODE_AL_BGR
    return _NODE_SAFE_BGR


def draw_spine_curve_bgr(image_bgr: np.ndarray,
                          spine: SilhouetteSpine | None,
                          curve_thick: int = 4,
                          marker_radius: int = 6,
                          ) -> np.ndarray:
    """Draw the trunk-axis spine plus the shoulder & pelvis girdle
    lines and a flexion / obliquity feedback label, in FULL image
    coordinates. The line, girdle and label are a SINGLE fixed colour
    on every worker; posture severity is read from the numeric
    `Flex / Obl` label. The only varying element is the vertebral
    NODES, coloured by the fixed NIOSH 1981 disc-compression
    thresholds (cyan < AL, orange AL–MPL, red ≥ MPL) when
    `spine.node_compression_N` is attached — so the disc level the
    flexion angle overloads pops out directly.

    No-op when `spine` is None so the caller can pass through the
    fall-back-to-v1 case transparently."""
    import cv2

    if spine is None:
        return image_bgr

    sev = _SPINE_BGR

    # Shoulder & pelvis girdle segments — make the twist / side-bend
    # visible: when these two lines are non-parallel the waist is
    # rotated or laterally bent.
    for seg in (spine.shoulder_xy, spine.hip_xy):
        if seg is None:
            continue
        a = (int(round(seg[0][0])), int(round(seg[0][1])))
        b = (int(round(seg[1][0])), int(round(seg[1][1])))
        cv2.line(image_bgr, a, b, _HALO_BGR, curve_thick + 4,
                 lineType=cv2.LINE_AA)
        cv2.line(image_bgr, a, b, _GIRDLE_BGR, curve_thick,
                 lineType=cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(image_bgr, p, marker_radius, _GIRDLE_BGR, -1,
                       lineType=cv2.LINE_AA)

    # Trunk axis (sacrum→T1), fixed cyan.
    bc = spine.back_curve_xy
    if bc is not None and len(bc) >= 2:
        pts = bc.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [pts], False, _HALO_BGR,
                      curve_thick + 4, lineType=cv2.LINE_AA)
        cv2.polylines(image_bgr, [pts], False, sev,
                      curve_thick, lineType=cv2.LINE_AA)
    # Vertebral nodes: cyan by default, but coloured by the fixed
    # NIOSH 1981 disc-compression thresholds where the per-level 3-D
    # load is available (cyan < AL, orange AL–MPL, red ≥ MPL). The
    # threshold is absolute, so the same load is the same colour on
    # every worker — only the high-load discs pop out.
    ncomp = spine.node_compression_N
    for i, xy in enumerate(spine.positions_xy):
        p = (int(round(xy[0])), int(round(xy[1])))
        node_c = sev
        big = marker_radius
        if ncomp is not None and i < len(ncomp):
            node_c = _node_bgr(float(ncomp[i]))
            if node_c is not _NODE_SAFE_BGR:        # AL/MPL → emphasise
                big = marker_radius + 2
        cv2.circle(image_bgr, p, big + 1, (255, 255, 255), -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, p, big, node_c, -1,
                   lineType=cv2.LINE_AA)

    # Feedback label near the thorax end of the trunk axis.
    if bc is not None and len(bc) >= 2:
        tip = bc[-1]
        org = (int(round(tip[0])) + 8, int(round(tip[1])) - 8)
        txt = (f"Flex {spine.trunk_flex_deg:.0f}deg  "
               f"Obl {spine.obliquity_deg:.0f}deg")
        cv2.putText(image_bgr, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _HALO_BGR, 4, cv2.LINE_AA)
        cv2.putText(image_bgr, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, sev, 2, cv2.LINE_AA)
    return image_bgr
