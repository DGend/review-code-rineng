"""Per-vertebral-disc compression and disc-herniation tolerance.

**Why this module exists**

§3.5.4 (`demographic_load`) adjusts per-*joint* moment capacity by sex
and age. §3.5.2 (`dense_spine`) already produces a per-*vertebral-
level* external static moment (Nm) from the RTMW3D **3-D** pose — so
forward trunk flexion *toward the camera* (the F6-Eldercare monocular-
depth case the 2-D four-keypoint feedback cannot see) **is** recovered
here, because `dense_spine.estimate_dense_spine` measures the trunk
axis from the 3-D keypoints, not the image plane.

What is still missing is the quantity that actually predicts
intervertebral-disc herniation: the **compressive force (N)** on each
disc, compared against that disc's compressive *tolerance*, which is
itself sex- and age-dependent. This module supplies exactly that and
unifies the §3.5.2 (dense spine) and §3.5.4 (sex/age) contributions.

**Model**

Chaffin's single-equivalent-muscle sagittal-plane static model
(Chaffin 1969; Chaffin & Andersson, *Occupational Biomechanics* 4th
ed., 2006):

    C_i = M_i / e  +  W_above_i · cos(θ)

where for vertebral level *i*:

* `M_i`  — external static moment at the disc (Nm), taken directly
  from `dense_spine` (body segments distal to the level, de Leva
  1996 mass fractions, computed from the **3-D** pose so depth-
  direction flexion is included);
* `e`    — erector-spinae extensor moment arm ≈ 0.05 m (Chaffin &
  Andersson 2006);
* `W_above_i` — gravitational weight of the body mass above the disc
  (`SPINE_DISTAL_MASS_FRAC` × body-mass × g);
* `θ`    — trunk flexion from vertical (3-D, from `dense_spine`).

The first term (muscle force balancing the flexion moment) dominates
in stooped postures and is the mechanism by which forward bending
drives L5/S1 disc-herniation risk.

**Tolerance (sex/age coupled)**

Ultimate compressive strength of the spinal motion segment for a
25-year-old male reference per level (Jäger & Luttmann 1989;
Brinckmann *et al.* 1988; Genaidy *et al.* 1993; the L5/S1 value
coincides with the NIOSH 1981 Maximum-Permissible-Limit of 6400 N).
Female endplate / vertebral compressive strength is ≈ 0.66 × male
(Jäger & Luttmann 1989), and strength declines with age following the
same multiplier used in `demographic_load` (Lindle 1997; Stenholm
2008). The personalised tolerance is therefore

    tol_i(sex, age) = ULT_25M[i] · sex_factor(sex) · age_mult(age, sex)

and the per-disc herniation-risk ratio is `C_i / tol_i`. The NIOSH
1981 Action Limit (3400 N) and Maximum Permissible Limit (6400 N) at
L5/S1 are retained as fixed population reference lines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ace_pose.demographic_load import (
    age_multiplier, DemographicProfile, DEFAULT_PROFILES, RISK_COLOR,
)
from ace_pose.dense_spine import (
    estimate_dense_spine, SpineSegments, SPINE_DISTAL_MASS_FRAC,
)


GRAVITY = 9.81
ERECTOR_MOMENT_ARM_M = 0.05          # Chaffin & Andersson 2006

# NIOSH 1981 compression criteria at L5/S1 (fixed population refs)
NIOSH_AL_N = 3400.0                   # Action Limit
NIOSH_MPL_N = 6400.0                  # Maximum Permissible Limit

# Ultimate compressive strength, 25-yo male reference (N), per level.
# Lumbar: Jäger & Luttmann 1989 / Brinckmann 1988 (L5/S1 == NIOSH MPL);
# thoracic segments tolerate more (smaller lever, rib-cage support).
DISC_ULT_STRENGTH_25M_N: dict[str, float] = {
    "sacrum":  8000.0,
    "L5/S1":   6400.0,
    "L3":      6000.0,
    "L1":      5600.0,
    "T12":     5800.0,
    "T6":      7000.0,
    "T1":      8000.0,
    "C7":      6000.0,
    "occiput": 6000.0,
}

# Female motion-segment compressive strength ≈ 0.66 × male
# (Jäger & Luttmann 1989).
SEX_DISC_FACTOR = {"M": 1.00, "F": 0.66}

# Lumbar levels that carry clinical disc-herniation risk.
HERNIATION_LEVELS = ["L5/S1", "L3", "L1", "T12"]

# Personalised-ratio risk bands (C_i / tol_i).
DISC_RISK_BANDS = [
    (0.33, "safe"),
    (0.66, "caution"),
    (1.00, "elevated"),
    (float("inf"), "surgical"),
]


def _sex1(sex: str) -> str:
    return sex.strip().upper()[:1] if sex else "M"


def trunk_flexion_deg_3d(kp25: np.ndarray, scores25: np.ndarray) -> float:
    """3-D trunk flexion (deg from vertical) — the depth-aware angle
    that the 2-D four-keypoint feedback cannot see.  Same convention
    as `dense_spine` (world-up = image-down +Y → [0,-1,0])."""
    L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
    NECK_IDX, PELVIS_IDX = 23, 24
    pelvis = 0.5 * (kp25[L_HIP] + kp25[R_HIP])
    if scores25[PELVIS_IDX] > 0:
        pelvis = kp25[PELVIS_IDX]
    thorax = 0.5 * (kp25[L_SHO] + kp25[R_SHO])
    if scores25[NECK_IDX] > 0:
        thorax = kp25[NECK_IDX]
    v = thorax - pelvis
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 0.0
    v = v / n
    world_up = np.array([0.0, -1.0, 0.0])
    cosang = float(np.clip(np.dot(v, world_up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def disc_compression_N(spine: SpineSegments,
                        body_mass_kg: float,
                        trunk_flex_deg: float) -> dict[str, float]:
    """Chaffin single-equivalent-muscle per-disc compression (N)."""
    cosf = math.cos(math.radians(trunk_flex_deg))
    out: dict[str, float] = {}
    for lvl, M in zip(spine.levels, spine.moments_Nm):
        w_above = (SPINE_DISTAL_MASS_FRAC.get(lvl, 0.0)
                   * body_mass_kg * GRAVITY)
        out[lvl] = float(M) / ERECTOR_MOMENT_ARM_M + w_above * cosf
    return out


def disc_tolerance_N(level: str, sex: str, age: int | float) -> float:
    base = DISC_ULT_STRENGTH_25M_N.get(level)
    if base is None:
        raise KeyError(f"level '{level}' has no reference strength")
    return (base * SEX_DISC_FACTOR.get(_sex1(sex), 1.0)
            * age_multiplier(age, sex))


def disc_risk_band(ratio: float) -> str:
    for thr, name in DISC_RISK_BANDS:
        if ratio < thr:
            return name
    return "surgical"


@dataclass
class DiscLevelResult:
    level: str
    compression_N: float
    tolerance_N: float
    ratio: float
    risk: str
    over_niosh_al: bool
    over_niosh_mpl: bool

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "compression_N": self.compression_N,
            "tolerance_N": self.tolerance_N,
            "ratio": self.ratio,
            "risk": self.risk,
            "over_NIOSH_AL": self.over_niosh_al,
            "over_NIOSH_MPL": self.over_niosh_mpl,
        }


@dataclass
class DiscAssessment:
    profile: DemographicProfile
    trunk_flex_deg: float
    levels: list[DiscLevelResult] = field(default_factory=list)


def assess_discs(kp25: np.ndarray,
                 scores25: np.ndarray,
                 body_mass_kg: float = 70.0,
                 length_scale_m: float = 1.0,
                 profiles: Sequence[DemographicProfile] = DEFAULT_PROFILES,
                 ) -> tuple[float, list[DiscAssessment]]:
    """Per-disc compression + sex/age-coupled herniation risk for a
    single 3-D major-25 pose, evaluated under each demographic
    profile.  Returns (trunk_flex_deg_3d, [DiscAssessment, …])."""
    spine = estimate_dense_spine(kp25, scores25,
                                 body_mass_kg=body_mass_kg,
                                 length_scale_m=length_scale_m)
    flex = trunk_flexion_deg_3d(kp25, scores25)
    comp = disc_compression_N(spine, body_mass_kg, flex)

    out: list[DiscAssessment] = []
    for prof in profiles:
        a = DiscAssessment(profile=prof, trunk_flex_deg=flex)
        for lvl in spine.levels:
            C = comp[lvl]
            tol = disc_tolerance_N(lvl, prof.sex, prof.age)
            ratio = C / max(tol, 1e-6)
            a.levels.append(DiscLevelResult(
                level=lvl, compression_N=C, tolerance_N=tol,
                ratio=ratio, risk=disc_risk_band(ratio),
                over_niosh_al=C >= NIOSH_AL_N,
                over_niosh_mpl=C >= NIOSH_MPL_N,
            ))
        out.append(a)
    return flex, out
