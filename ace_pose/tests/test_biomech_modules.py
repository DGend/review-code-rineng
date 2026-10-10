"""Closed-form regression / invariant tests for the deterministic
biomechanical-refinement modules used in §3.5.2–3.5.5 / §4.5:

  * ace_pose.demographic_load  (sex/age effective-load adjustment)
  * ace_pose.disc_load         (Chaffin per-disc compression + tolerance)
  * ace_pose.dense_spine       (9-level anatomical interpolation)

These modules are deterministic and parameter-free, so every test
asserts an exact closed-form value (computed by hand from the model
equation) or a structural invariant — no model inference, no fixtures,
no randomness.  They lock the numbers reported in the manuscript so a
future refactor cannot silently move them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.demographic_load import (
    age_multiplier, effective_capacity_nm, effective_load_rate,
    risk_category, CAPACITY_25M_NM, CAPACITY_25F_NM,
)
from ace_pose.disc_load import (
    disc_tolerance_N, disc_risk_band, disc_compression_N,
    trunk_flexion_deg_3d, NIOSH_AL_N, NIOSH_MPL_N,
    ERECTOR_MOMENT_ARM_M, SEX_DISC_FACTOR, GRAVITY, HERNIATION_LEVELS,
    DISC_ULT_STRENGTH_25M_N,
)
from ace_pose.dense_spine import (
    estimate_dense_spine, SPINE_LEVELS, SPINE_DISTAL_MASS_FRAC,
)

_LEVELS = ["sacrum", "L5/S1", "L3", "L1", "T12", "T6", "T1", "C7",
           "occiput"]


def _synth_kp25(thorax_xyz):
    """A minimal major-25 pose: pelvis at origin, thorax at the given
    offset, nose slightly above-and-anterior. scores all = 1 so the
    estimators use the derived neck(23)/pelvis(24) directly."""
    kp = np.zeros((25, 3), dtype=float)
    kp[24] = (0.0, 0.0, 0.0)                       # PELVIS_IDX
    kp[23] = np.asarray(thorax_xyz, dtype=float)   # NECK_IDX
    kp[0] = kp[23] + np.array([0.0, -0.25, 0.10])  # NOSE (anterior ref)
    kp[5] = kp[23] + np.array([0.20, 0.0, 0.0])    # L_SHO
    kp[6] = kp[23] + np.array([-0.20, 0.0, 0.0])   # R_SHO
    kp[11] = np.array([0.15, 0.0, 0.0])            # L_HIP
    kp[12] = np.array([-0.15, 0.0, 0.0])           # R_HIP
    sc = np.ones(25, dtype=float)
    return kp, sc


# --------------------------------------------------------------------
class TestDemographicLoad:
    def test_age_multiplier_closed_form(self):
        assert age_multiplier(25, "M") == pytest.approx(1.00)
        assert age_multiplier(25, "F") == pytest.approx(1.00)
        # 1 %/yr decline 25→50
        assert age_multiplier(40, "M") == pytest.approx(0.85)
        # post-50 acceleration + post-menopausal 0.92 factor for F
        assert age_multiplier(55, "F") == pytest.approx(0.675 * 0.92)
        assert age_multiplier(65, "F") == pytest.approx(0.525 * 0.92)
        # 30 % capacity floor
        assert age_multiplier(200, "M") == pytest.approx(0.30)

    def test_female_capacity_ratio(self):
        # the manuscript's 0.66 female L5/S1 moment-capacity factor
        assert (CAPACITY_25F_NM["L5S1"] / CAPACITY_25M_NM["L5S1"]
                == pytest.approx(0.66, abs=1e-9))

    def test_effective_capacity_and_rate(self):
        assert effective_capacity_nm("L5S1", "M", 25) == pytest.approx(250.0)
        assert effective_capacity_nm("L5S1", "F", 25) == pytest.approx(165.0)
        assert effective_capacity_nm("L5S1", "M", 40) == pytest.approx(212.5)
        # 125 Nm against a 250 Nm capacity = 0.5 load rate
        assert effective_load_rate(125.0, "L5S1", "M", 25) == pytest.approx(0.5)

    def test_risk_category_band_edges(self):
        assert risk_category(0.0) == "safe"
        assert risk_category(0.299) == "safe"
        assert risk_category(0.30) == "caution"
        assert risk_category(0.50) == "elevated"
        assert risk_category(0.799) == "elevated"
        assert risk_category(0.80) == "surgical"
        assert risk_category(5.0) == "surgical"


# --------------------------------------------------------------------
class TestDiscLoad:
    def test_constants_locked(self):
        assert NIOSH_AL_N == 3400.0
        assert NIOSH_MPL_N == 6400.0
        assert ERECTOR_MOMENT_ARM_M == 0.05
        assert SEX_DISC_FACTOR == {"M": 1.00, "F": 0.66}
        assert GRAVITY == pytest.approx(9.81)
        assert HERNIATION_LEVELS == ["L5/S1", "L3", "L1", "T12"]
        # L5/S1 reference strength coincides with the NIOSH MPL
        assert DISC_ULT_STRENGTH_25M_N["L5/S1"] == NIOSH_MPL_N

    def test_disc_tolerance_closed_form(self):
        assert disc_tolerance_N("L5/S1", "M", 25) == pytest.approx(6400.0)
        assert disc_tolerance_N("L5/S1", "F", 25) == pytest.approx(4224.0)
        assert disc_tolerance_N("L5/S1", "M", 40) == pytest.approx(5440.0)

    def test_disc_risk_band_edges(self):
        assert disc_risk_band(0.0) == "safe"
        assert disc_risk_band(0.329) == "safe"
        assert disc_risk_band(0.33) == "caution"
        assert disc_risk_band(0.66) == "elevated"
        assert disc_risk_band(0.999) == "elevated"
        assert disc_risk_band(1.00) == "surgical"
        assert disc_risk_band(3.2) == "surgical"

    def test_trunk_flexion_3d_closed_form(self):
        # thorax straight up (image-down convention) → 0°
        kp, sc = _synth_kp25((0.0, -1.0, 0.0))
        assert trunk_flexion_deg_3d(kp, sc) == pytest.approx(0.0, abs=1e-6)
        # thorax horizontal → 90°
        kp, sc = _synth_kp25((1.0, 0.0, 0.0))
        assert trunk_flexion_deg_3d(kp, sc) == pytest.approx(90.0, abs=1e-6)
        # 45° lean
        kp, sc = _synth_kp25((math.sin(math.radians(45)),
                              -math.cos(math.radians(45)), 0.0))
        assert trunk_flexion_deg_3d(kp, sc) == pytest.approx(45.0, abs=1e-6)

    def test_compression_matches_chaffin_equation(self):
        """C_i = M_i/e + W_above_i·cos(theta) — exact, per level."""
        kp, sc = _synth_kp25((0.3, -1.0, 0.0))
        spine = estimate_dense_spine(kp, sc, body_mass_kg=70.0,
                                     length_scale_m=1.0)
        flex = 35.0
        comp = disc_compression_N(spine, 70.0, flex)
        cosf = math.cos(math.radians(flex))
        for i, lvl in enumerate(spine.levels):
            w_above = SPINE_DISTAL_MASS_FRAC.get(lvl, 0.0) * 70.0 * GRAVITY
            expected = float(spine.moments_Nm[i]) / ERECTOR_MOMENT_ARM_M \
                + w_above * cosf
            assert comp[lvl] == pytest.approx(expected, rel=1e-9, abs=1e-6)


# --------------------------------------------------------------------
class TestDenseSpine:
    def test_levels_structure(self):
        assert [n for n, _, _ in SPINE_LEVELS] == _LEVELS
        assert set(_LEVELS).issubset(set(SPINE_DISTAL_MASS_FRAC))
        # distal mass decreases monotonically up the spine
        fr = [SPINE_DISTAL_MASS_FRAC[l] for l in _LEVELS]
        assert fr == sorted(fr, reverse=True)
        assert fr[0] == pytest.approx(0.66) and fr[-1] == pytest.approx(0.06)

    def test_estimate_shapes_and_determinism(self):
        kp, sc = _synth_kp25((0.0, -1.0, 0.0))
        a = estimate_dense_spine(kp, sc, body_mass_kg=70.0,
                                 length_scale_m=1.0)
        b = estimate_dense_spine(kp, sc, body_mass_kg=70.0,
                                 length_scale_m=1.0)
        assert a.levels == _LEVELS
        assert a.positions.shape == (9, 3)
        assert a.moments_Nm.shape == (9,)
        assert np.array_equal(a.positions, b.positions)      # deterministic
        assert np.array_equal(a.moments_Nm, b.moments_Nm)
        assert np.all(np.isfinite(a.moments_Nm))
        assert np.all(a.moments_Nm >= 0.0)

    def test_moment_monotone_down_the_spine(self):
        kp, sc = _synth_kp25((0.25, -1.0, 0.0))
        s = estimate_dense_spine(kp, sc, body_mass_kg=70.0,
                                 length_scale_m=1.0)
        m = s.moments_Nm
        # external moment is largest at the sacrum and non-increasing
        # toward the occiput (distal mass shrinks upward)
        assert np.all(np.diff(m) <= 1e-9)
        assert m[0] >= m[-1]

    def test_moment_scales_linearly(self):
        kp, sc = _synth_kp25((0.3, -1.0, 0.0))
        base = estimate_dense_spine(kp, sc, 70.0, 1.0).moments_Nm
        mass2 = estimate_dense_spine(kp, sc, 140.0, 1.0).moments_Nm
        len2 = estimate_dense_spine(kp, sc, 70.0, 2.0).moments_Nm
        idx = base > 1e-6
        assert np.allclose(mass2[idx] / base[idx], 2.0, rtol=1e-6)
        assert np.allclose(len2[idx] / base[idx], 2.0, rtol=1e-6)

    def test_flex_is_bounded_template_and_responds_to_lean(self):
        """DSE v1 is an *anatomical interpolation*: even an upright
        trunk carries the bounded lordosis/kyphosis template curvature
        (it is NOT zero — this is the documented template behaviour,
        §4.5.2). Leaning the trunk must increase the mean per-level
        flexion deterministically."""
        kp_up, sc = _synth_kp25((0.0, -1.0, 0.0))
        s_up = estimate_dense_spine(kp_up, sc, 70.0, 1.0)
        s_up2 = estimate_dense_spine(kp_up, sc, 70.0, 1.0)
        assert np.array_equal(s_up.flex_deg, s_up2.flex_deg)   # deterministic
        assert np.all(np.isfinite(s_up.flex_deg))
        # upright → only the bounded anatomical template (no runaway)
        assert 0.0 <= float(np.max(np.abs(s_up.flex_deg))) < 40.0
        # a strongly forward-leaned trunk raises the mean flexion
        kp_bent, _ = _synth_kp25((0.9, -0.6, 0.0))
        s_bent = estimate_dense_spine(kp_bent, sc, 70.0, 1.0)
        assert float(np.mean(np.abs(s_bent.flex_deg))) > \
            float(np.mean(np.abs(s_up.flex_deg)))
