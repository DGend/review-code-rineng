"""Sex- and age-specific load-capacity adjustment for the BIOERA
ergonomic-risk pipeline.

**Why this module exists**

The three seminal ergonomic-risk methods used in this paper —
OWAS (Karhu et al., 1977), RULA (McAtamney & Corlett, 1993) and REBA
(Hignett & McAtamney, 2000) — were developed using predominantly
male, working-age (25-40) subjects and report **a single action
category regardless of the worker's sex or age**. The biomechanical
literature, however, documents large and well-replicated sex / age
differences in:

* **Lumbar moment capacity**  — female peak L5/S1 moment is ≈ 65 %
  of male peak (Marras et al., 2002, *Spine*).
* **Hip and lower-extremity capacity** — female peak hip extension
  ≈ 70 % of male (Chaffin & Andersson, *Occupational Biomechanics*
  4th ed., 2006, Ch. 4).
* **Shoulder / upper-extremity capacity** — female peak shoulder
  abduction ≈ 56 % of male (Astrand & Rodahl, *Textbook of Work
  Physiology*).
* **Age-related strength decline** — overall muscle strength falls
  ≈ 1 % per year after age 25-30, accelerates to ≈ 1.5 % / year
  after 50 and ≈ 2 % / year after 65 (Lindle et al., 1997, *J.
  Appl. Physiol.*; Janssen et al., 2000, *J. Appl. Physiol.*).
* **Post-menopausal bone-density** — additional ≈ 8 % strength
  reduction for women age ≥ 50 (Stenholm et al., 2008, *Curr.
  Opin. Clin. Nutr. Metab. Care*).

Consequently the *same* 60° trunk flexion carrying 5 kg is "moderate
risk" for a 25-year-old male (within 35 % of his L5/S1 capacity) but
"surgical risk" for a 55-year-old female (≈ 82 % of her capacity) —
yet OWAS / RULA / REBA assign the *same* action category to both.

**What this module does**

Given a per-joint observed external moment (output of
`ace_pose.biomechanics.compute_static_moments`) plus a worker's sex
and age, this module returns the **effective load rate** — the
fraction of the worker's individual maximum static-moment capacity
that the observed pose consumes — and a four-level risk category:

  * `< 30 %` → safe
  * `30–50 %` → caution
  * `50–80 %` → elevated
  * `> 80 %` → surgical-risk

The result is **directly clinically actionable**: a 55-year-old
female bathing-task caregiver with an effective L5/S1 load rate of
72 % is in the *elevated* category that occupational-health teams
can target with workstation-redesign or rotation, whereas the
underlying OWAS action category may report only "1 — no action
needed".

This is, to our knowledge, the first occupational-health pipeline
that fuses pose-derived per-joint moments with sex- and age-adjusted
capacity tables. The standard ergonomic-risk methods (OWAS / RULA /
REBA) do **not** carry such an adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# -------------------------------------------------------------------
# Reference capacity at the demographic optimum (25-year-old).
# Values are peak static external moment capacity (Nm) per joint.
# -------------------------------------------------------------------
# Sources:
#   L5S1   — Marras 2002, Spine (LMM EMG-assisted model, n = 76)
#   hip    — Chaffin & Andersson 2006 Ch. 4
#   knee   — Frontera 1991, isokinetic peak torque
#   shoulder — Astrand & Rodahl 1986
CAPACITY_25M_NM: dict[str, float] = {
    "L5S1":     250.0,
    "hip":      160.0,
    "knee":     200.0,
    "shoulder":  50.0,
}

CAPACITY_25F_NM: dict[str, float] = {
    "L5S1":     165.0,        # 0.66 × male
    "hip":      112.0,        # 0.70 × male
    "knee":     160.0,        # 0.80 × male
    "shoulder":  28.0,        # 0.56 × male
}


def _sex_capacity_table(sex: str) -> dict[str, float]:
    s = sex.strip().upper()[:1] if sex else "M"
    return CAPACITY_25F_NM if s == "F" else CAPACITY_25M_NM


# -------------------------------------------------------------------
# Age decay
# -------------------------------------------------------------------
def age_multiplier(age: int | float, sex: str = "M") -> float:
    """Capacity multiplier as a function of age.

    Sources:
      * Lindle et al. 1997 — strength declines ~ 1 % / yr after 25-30
      * Stenholm et al. 2008 — post-50 acceleration
      * Janssen et al. 2000 — age-related muscle mass distribution

    Additional 8 % factor for women age ≥ 50 (post-menopausal bone-
    density-related load tolerance) per Stenholm 2008."""
    age = float(age)
    if age <= 25:
        m = 1.00
    elif age <= 50:
        m = 1.00 - 0.010 * (age - 25)             # 1 % per year
    elif age <= 65:
        m = (1.00 - 0.010 * 25) - 0.015 * (age - 50)
    else:
        m_65 = 1.00 - 0.010 * 25 - 0.015 * 15
        m = m_65 - 0.020 * (age - 65)

    s = sex.strip().upper()[:1] if sex else "M"
    if s == "F" and age >= 50:
        m *= 0.92                                  # post-menopausal extra
    return max(m, 0.30)                            # floor at 30 % capacity


# -------------------------------------------------------------------
# Per-joint adjusted capacity + effective load rate
# -------------------------------------------------------------------
def effective_capacity_nm(joint: str, sex: str, age: int | float) -> float:
    base = _sex_capacity_table(sex).get(joint)
    if base is None:
        raise KeyError(f"joint '{joint}' has no reference capacity")
    return base * age_multiplier(age, sex)


def effective_load_rate(observed_nm: float, joint: str,
                         sex: str, age: int | float) -> float:
    """Fraction of the worker's individual maximum static-moment
    capacity consumed by the observed pose at this joint."""
    cap = effective_capacity_nm(joint, sex, age)
    return float(observed_nm) / max(cap, 1e-6)


# Risk categories
RISK_BANDS = [
    (0.30, "safe"),
    (0.50, "caution"),
    (0.80, "elevated"),
    (float("inf"), "surgical"),
]
RISK_COLOR = {
    "safe":      "#16a34a",
    "caution":   "#facc15",
    "elevated":  "#ea580c",
    "surgical":  "#dc2626",
}


def risk_category(load_rate: float) -> str:
    for thr, name in RISK_BANDS:
        if load_rate < thr:
            return name
    return "surgical"


# -------------------------------------------------------------------
# Demographic-scenario analysis
# -------------------------------------------------------------------
@dataclass
class DemographicProfile:
    label: str
    sex: str
    age: int


DEFAULT_PROFILES: list[DemographicProfile] = [
    DemographicProfile("25 M",  "M", 25),
    DemographicProfile("40 M",  "M", 40),
    DemographicProfile("55 M",  "M", 55),
    DemographicProfile("25 F",  "F", 25),
    DemographicProfile("40 F",  "F", 40),
    DemographicProfile("55 F",  "F", 55),
    DemographicProfile("65 F",  "F", 65),
]


@dataclass
class JointDemographicAssessment:
    joint: str
    observed_moment_nm: float
    capacity_nm: float
    load_rate: float
    risk: str

    def to_dict(self) -> dict:
        return {
            "joint":       self.joint,
            "moment_Nm":   self.observed_moment_nm,
            "capacity_Nm": self.capacity_nm,
            "load_rate":   self.load_rate,
            "risk":        self.risk,
        }


@dataclass
class DemographicAssessment:
    profile: DemographicProfile
    joints: list[JointDemographicAssessment] = field(default_factory=list)

    def add(self, joint: str, observed_nm: float):
        cap = effective_capacity_nm(joint, self.profile.sex,
                                     self.profile.age)
        lr  = observed_nm / max(cap, 1e-6)
        self.joints.append(JointDemographicAssessment(
            joint=joint, observed_moment_nm=observed_nm,
            capacity_nm=cap, load_rate=lr, risk=risk_category(lr),
        ))


def assess_pose(moments_nm: dict[str, float],
                 profiles: Sequence[DemographicProfile] = DEFAULT_PROFILES
                 ) -> list[DemographicAssessment]:
    """For a single pose's per-joint moments, compute the effective
    load rate under each demographic profile."""
    out: list[DemographicAssessment] = []
    for prof in profiles:
        a = DemographicAssessment(profile=prof)
        for jname, mom in moments_nm.items():
            if jname not in CAPACITY_25M_NM:
                continue
            a.add(jname, mom)
        out.append(a)
    return out
