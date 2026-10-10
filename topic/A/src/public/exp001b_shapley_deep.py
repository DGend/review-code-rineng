# PWW-DISCLOSURE: public
"""
exp-001b-shapley-deep

Follow-up to exp-001-shapley-axioms addressing /pw-verify 2026-08-08
domain-reviewer major issues:

  M3.a additivity axiom (not tested in exp-001) - constructed pair of
       cooperative games v1, v2 over OWAS/RULA/REBA evaluators; verify
       phi(v1 + v2) == phi(v1) + phi(v2).
  M3.b monotonicity vs OWAS/RULA/REBA lookup structure - for each
       scheme, sample an axis vector, increase one axis by 1, check
       evaluator(v') >= evaluator(v) (score must not decrease as an
       ergonomic risk axis worsens). Report violation rate.
  T18  symmetry axiom in its correct cooperative-game form - build a
       synthetic 4-axis game v(S) that is invariant under swapping
       axes {0,1}; verify shap[0] == shap[1].
  minor.a  formal dummy - append an (N+1)-th axis that the evaluator
           literally ignores; verify shap of that axis == 0 exact.
  minor.b  multi-seed x corner-case coverage - 5 seeds x 200 trials
           per scheme plus deterministic all-min and all-max corners.
  minor.c  residual distribution p50 / p99 and relative residual.
  minor.d  independent Shapley cross-check via permutation Monte Carlo
           (no shap package required); report max disagreement.

Records only facts. No pass/fail declaration.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

TOPIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOPIC_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.xai import (  # type: ignore
    OWAS_AXES,
    RULA_AXES,
    REBA_AXES,
    _eval_owas,
    _eval_rula,
    _eval_reba,
    _axis_min,
    shapley_attribution,
)


TOL_ABS = 1e-9
SCHEMES = [
    ("OWAS", OWAS_AXES, _eval_owas),
    ("RULA", RULA_AXES, _eval_rula),
    ("REBA", REBA_AXES, _eval_reba),
]


# ------------------------------------------------------------------
# generic Shapley over a callable evaluator on a value dict
# ------------------------------------------------------------------

def shapley_exact(values, axes, evaluator):
    """Reference: same convention as ace_pose.xai.shapley_attribution
    (min-baseline v(S) = evaluator(actual for axes in S, min otherwise))."""
    return shapley_attribution(values, axes, evaluator)


def shapley_perm_mc(values, axes, evaluator, n_perm, rng):
    """Independent implementation: permutation Monte Carlo.
    phi_i = E_pi[v(pred_pi(i) + {i}) - v(pred_pi(i))]."""
    axis_names = list(axes.keys())
    mins = _axis_min(axes)
    def v_of(subset):
        cfg = dict(mins)
        for a in subset:
            cfg[a] = values[a]
        return float(evaluator(cfg))
    acc = {a: 0.0 for a in axis_names}
    for _ in range(n_perm):
        perm = axis_names[:]
        rng.shuffle(perm)
        prefix = []
        v_prev = v_of(tuple(prefix))
        for a in perm:
            prefix.append(a)
            v_now = v_of(tuple(prefix))
            acc[a] += v_now - v_prev
            v_prev = v_now
    return {a: acc[a] / n_perm for a in axis_names}


def v_grand_minus_empty(values, axes, evaluator):
    mins = _axis_min(axes)
    return float(evaluator(values)) - float(evaluator(mins))


# ------------------------------------------------------------------
# checks
# ------------------------------------------------------------------

def check_efficiency_deep(values, axes, evaluator):
    shap = shapley_exact(values, axes, evaluator)
    g = v_grand_minus_empty(values, axes, evaluator)
    residual = sum(shap.values()) - g
    rel = abs(residual) / max(abs(g), 1.0)
    return residual, rel


def check_additivity(values, axes, ev1, ev2):
    """Two games v1, v2 (same axis set, same min-baseline). Check
    phi(v1 + v2) == phi(v1) + phi(v2)."""
    def ev_sum(cfg):
        return ev1(cfg) + ev2(cfg)
    phi1 = shapley_exact(values, axes, ev1)
    phi2 = shapley_exact(values, axes, ev2)
    phi_sum = shapley_exact(values, axes, ev_sum)
    max_dev = max(abs(phi_sum[a] - (phi1[a] + phi2[a])) for a in axes)
    return max_dev


def check_monotonicity(values, axes, evaluator):
    """For each axis, increment by 1 (if room), check evaluator does not
    decrease. Returns list of (axis, before, after, delta)."""
    violations = []
    base = float(evaluator(values))
    for a, (lo, hi) in axes.items():
        if values[a] < hi:
            v2 = dict(values); v2[a] = values[a] + 1
            after = float(evaluator(v2))
            if after < base - 1e-12:
                violations.append({"axis": a, "before": base, "after": after,
                                   "delta": after - base, "values": dict(v2)})
    return violations


def check_formal_dummy(values, axes, evaluator):
    """Append an (N+1)-th synthetic axis 'dummy_z' that the evaluator
    literally ignores; verify shap of that axis is exactly 0."""
    extended_axes = dict(axes)
    extended_axes["dummy_z"] = (1, 4)
    def ev_ignore(cfg):
        c2 = {k: v for k, v in cfg.items() if k != "dummy_z"}
        return evaluator(c2)
    ext_values = dict(values); ext_values["dummy_z"] = 3
    shap = shapley_exact(ext_values, extended_axes, ev_ignore)
    return shap["dummy_z"]


def check_symmetry_constructed(rng):
    """Build a synthetic 4-axis game v(S) that is invariant under
    swapping axes A and B, and check phi[A] == phi[B]."""
    axes = {"A": (0, 5), "B": (0, 5), "C": (0, 5), "D": (0, 5)}
    values = {"A": rng.randint(0, 5), "B": rng.randint(0, 5),
              "C": rng.randint(0, 5), "D": rng.randint(0, 5)}
    values["B"] = values["A"]  # equal values so game invariance implies phi equality
    def ev(cfg):
        # Symmetric in A and B by construction: uses A+B and A*B only.
        return (cfg["A"] + cfg["B"]) * 2 + cfg["A"] * cfg["B"] + cfg["C"] * 3 + cfg["D"]
    phi = shapley_exact(values, axes, ev)
    return abs(phi["A"] - phi["B"]), phi


def check_cross_impl(values, axes, evaluator, n_perm, rng):
    """Compare ace_pose exact Shapley to an independent permutation-MC
    Shapley; report max per-axis disagreement."""
    exact = shapley_exact(values, axes, evaluator)
    mc = shapley_perm_mc(values, axes, evaluator, n_perm, rng)
    return max(abs(exact[a] - mc[a]) for a in axes), exact, mc


# ------------------------------------------------------------------
# runners
# ------------------------------------------------------------------

def run_scheme(name, axes, evaluator, seeds, n_trials, mc_perm):
    all_eff_residuals = []
    all_eff_relative = []
    all_dummy_shap = []
    mono_violation_count = 0
    mono_total_checks = 0
    cross_max_disagree = []
    for seed in seeds:
        rng = random.Random(seed)
        # corner cases first (deterministic)
        for corner_kind in ("min", "max"):
            v = {a: (lo if corner_kind == "min" else hi) for a, (lo, hi) in axes.items()}
            r, rel = check_efficiency_deep(v, axes, evaluator)
            all_eff_residuals.append(abs(r)); all_eff_relative.append(rel)
            all_dummy_shap.append(abs(check_formal_dummy(v, axes, evaluator)))
        for _ in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            r, rel = check_efficiency_deep(values, axes, evaluator)
            all_eff_residuals.append(abs(r)); all_eff_relative.append(rel)
            all_dummy_shap.append(abs(check_formal_dummy(values, axes, evaluator)))
            mono_viol = check_monotonicity(values, axes, evaluator)
            mono_violation_count += len(mono_viol)
            mono_total_checks += sum(1 for a, (lo, hi) in axes.items() if values[a] < hi)
            cross_rng = random.Random(seed * 7919 + 1)
            max_dis, _, _ = check_cross_impl(values, axes, evaluator, mc_perm, cross_rng)
            cross_max_disagree.append(max_dis)
    return {
        "eff_residual_max": max(all_eff_residuals),
        "eff_residual_p99": statistics.quantiles(all_eff_residuals, n=100)[98] if len(all_eff_residuals) >= 100 else max(all_eff_residuals),
        "eff_residual_p50": statistics.median(all_eff_residuals),
        "eff_relative_max": max(all_eff_relative),
        "dummy_shap_max": max(all_dummy_shap),
        "monotonicity_violation_rate": mono_violation_count / max(mono_total_checks, 1),
        "monotonicity_violations": mono_violation_count,
        "monotonicity_checks": mono_total_checks,
        "cross_impl_max_disagree": max(cross_max_disagree),
        "cross_impl_median_disagree": statistics.median(cross_max_disagree),
        "n_trials_effective": len(all_eff_residuals),
    }


def run_additivity(axes, ev1, ev2, seeds, n_trials):
    max_devs = []
    for seed in seeds:
        rng = random.Random(seed * 31 + 3)
        for _ in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            max_devs.append(check_additivity(values, axes, ev1, ev2))
    return {"additivity_max_dev": max(max_devs),
            "additivity_median_dev": statistics.median(max_devs),
            "n_pairs": len(max_devs)}


def main():
    seeds = [20260808, 20260809, 20260810, 20260811, 20260812]
    n_trials = 200
    mc_perm = 500

    results = {"config": {"seeds": seeds, "n_trials_per_seed": n_trials,
                          "mc_permutations": mc_perm, "tolerance_abs": TOL_ABS}}

    for name, axes, ev in SCHEMES:
        results[name] = run_scheme(name, axes, ev, seeds, n_trials, mc_perm)

    # Additivity checks: pair each scheme with a synthetic co-game that
    # scales the evaluator's output by a random weight. Use axes of the
    # first scheme (OWAS) for the pair; a scheme against itself scaled.
    def make_scaled(ev, k):
        def scaled(cfg): return k * ev(cfg)
        return scaled

    add_results = {}
    for name, axes, ev in SCHEMES:
        ev2 = make_scaled(ev, 3.5)
        add_results[name] = run_additivity(axes, ev, ev2, seeds[:2], 100)
    results["additivity"] = add_results

    # Constructed symmetry (T18)
    sym_rng = random.Random(sum(seeds))
    sym_max = 0.0
    sym_probes = []
    for _ in range(50):
        dev, phi = check_symmetry_constructed(sym_rng)
        sym_max = max(sym_max, dev)
        sym_probes.append({"dev": dev, "phi": phi})
    results["symmetry_constructed"] = {
        "max_dev_between_symmetric_players": sym_max,
        "median_dev": statistics.median([p["dev"] for p in sym_probes]),
        "n_probes": len(sym_probes),
        "example_phi": sym_probes[0]["phi"],
    }

    metrics = {
        # T16 efficiency
        "T16_efficiency_owas_max_residual": results["OWAS"]["eff_residual_max"],
        "T16_efficiency_rula_max_residual": results["RULA"]["eff_residual_max"],
        "T16_efficiency_reba_max_residual": results["REBA"]["eff_residual_max"],
        "T16_efficiency_max_relative_residual_owas": results["OWAS"]["eff_relative_max"],
        "T16_efficiency_max_relative_residual_rula": results["RULA"]["eff_relative_max"],
        "T16_efficiency_max_relative_residual_reba": results["REBA"]["eff_relative_max"],
        # T17 dummy (formal, N+1 axis)
        "T17_formal_dummy_owas_max_shap": results["OWAS"]["dummy_shap_max"],
        "T17_formal_dummy_rula_max_shap": results["RULA"]["dummy_shap_max"],
        "T17_formal_dummy_reba_max_shap": results["REBA"]["dummy_shap_max"],
        # T18 symmetry (constructed interchangeable-player game)
        "T18_symmetry_constructed_max_dev": results["symmetry_constructed"]["max_dev_between_symmetric_players"],
        "T18_symmetry_constructed_median_dev": results["symmetry_constructed"]["median_dev"],
        "T18_symmetry_scope_note": "verified on synthetic 4-axis game invariant under swap of A and B; OWAS/RULA/REBA axes are NOT interchangeable so axis-permutation symmetry is not claimed",
        # additivity (new axiom)
        "additivity_max_dev_owas": add_results["OWAS"]["additivity_max_dev"],
        "additivity_max_dev_rula": add_results["RULA"]["additivity_max_dev"],
        "additivity_max_dev_reba": add_results["REBA"]["additivity_max_dev"],
        # monotonicity vs standards (domain M3.b)
        "monotonicity_violation_rate_owas": results["OWAS"]["monotonicity_violation_rate"],
        "monotonicity_violation_rate_rula": results["RULA"]["monotonicity_violation_rate"],
        "monotonicity_violation_rate_reba": results["REBA"]["monotonicity_violation_rate"],
        "monotonicity_violations_owas": results["OWAS"]["monotonicity_violations"],
        "monotonicity_violations_rula": results["RULA"]["monotonicity_violations"],
        "monotonicity_violations_reba": results["REBA"]["monotonicity_violations"],
        # residual distribution
        "residual_p50_owas": results["OWAS"]["eff_residual_p50"],
        "residual_p50_rula": results["RULA"]["eff_residual_p50"],
        "residual_p50_reba": results["REBA"]["eff_residual_p50"],
        "residual_p99_owas": results["OWAS"]["eff_residual_p99"],
        "residual_p99_rula": results["RULA"]["eff_residual_p99"],
        "residual_p99_reba": results["REBA"]["eff_residual_p99"],
        # cross-implementation (permutation MC vs exact)
        "cross_impl_max_disagree_owas": results["OWAS"]["cross_impl_max_disagree"],
        "cross_impl_max_disagree_rula": results["RULA"]["cross_impl_max_disagree"],
        "cross_impl_max_disagree_reba": results["REBA"]["cross_impl_max_disagree"],
        # coverage
        "n_effective_trials_owas": results["OWAS"]["n_trials_effective"],
        "n_effective_trials_rula": results["RULA"]["n_trials_effective"],
        "n_effective_trials_reba": results["REBA"]["n_trials_effective"],
        "seeds": seeds,
    }

    result_dir = Path(os.environ.get("PWW_RESULT_DIR", TOPIC_ROOT / "result" / "exp-001b-shapley-deep"))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (result_dir / "raw.json").write_text(json.dumps(results, indent=2, default=str))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
