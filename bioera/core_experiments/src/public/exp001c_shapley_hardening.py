# PWW-DISCLOSURE: public
"""
exp-001c-shapley-hardening

Addresses the code-level minors surfaced by the two /pw-verify rounds on
exp-001 and exp-001b (see paper/reviews/peer/20260807T162222Z,
20260807T164417Z, 20260808T081357Z).

Adds:
  1  cross-impl MC RNG advanced per trial (independence)
  2  additivity co-game structurally independent of v1 (random-coeff
     linear game over the same axis set, not a scalar multiple)
  3  symmetry probe with distinct player values on a game that is
     structurally swap-invariant in {A, B}
  4  monotonicity_checks_* denominators exported as metric fields
  5  per-axis OWAS violation breakdown
  6  exhaustive OWAS 4-tuple enumeration of all axis-neighbours
     (increment AND decrement); returns concrete offending tuples
  7  decrement direction added to random-sample monotonicity + starting
     value stratification
  8  Wilson 95% CI for the pooled OWAS 345/2947 rate + per-seed rate
  9  MC 5000-permutation sanity subsample confirming sqrt(10) shrinkage
  10 dummy range / value sweep to confirm phi==0 invariance
  11 OWAS efficiency residual distribution replaces p99 with
     exact_zero_fraction (more informative when >=99% are exact zero)
  12 corners explicitly excluded from monotonicity denominator, noted

Records only facts. No pass/fail. `/pw-verify` judges.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

TOPIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOPIC_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.xai import (  # type: ignore
    OWAS_AXES, RULA_AXES, REBA_AXES,
    _eval_owas, _eval_rula, _eval_reba,
    _axis_min, shapley_attribution,
)

TOL_ABS = 1e-9
SCHEMES = [
    ("OWAS", OWAS_AXES, _eval_owas),
    ("RULA", RULA_AXES, _eval_rula),
    ("REBA", REBA_AXES, _eval_reba),
]


def shapley_exact(values, axes, evaluator):
    return shapley_attribution(values, axes, evaluator)


def shapley_perm_mc(values, axes, evaluator, n_perm, rng):
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
# minor #2: structurally independent co-game
# ------------------------------------------------------------------

def make_independent_cogame(axes, seed):
    """A random-coefficient linear game over the axis set. For subset
    S, v(S) = sum_a in S w[a] * (values[a] - min[a]) + bias. Structurally
    independent of the original evaluator."""
    rng = random.Random(seed)
    w = {a: rng.uniform(-2.0, 2.0) for a in axes}
    bias = rng.uniform(-1.0, 1.0)
    mins = _axis_min(axes)
    def ev(cfg):
        s = bias
        for a, wa in w.items():
            s += wa * (cfg[a] - mins[a])
        return s
    return ev


# ------------------------------------------------------------------
# minor #3: symmetry with distinct A,B values
# ------------------------------------------------------------------

def check_symmetry_distinct_values(rng):
    """v is symmetric in {A,B}: v(S) uses (A+B) and (A*B) only.
    Give A and B distinct values. If A,B are interchangeable players
    in v (i.e. v(S+A) = v(S+B) for all S not containing either), the
    Shapley axiom guarantees phi[A] = phi[B] REGARDLESS of the numeric
    values. The (A+B) and (A*B) form ensures interchangeability."""
    axes = {"A": (0, 5), "B": (0, 5), "C": (0, 5), "D": (0, 5)}
    values = {"A": rng.randint(0, 5), "B": rng.randint(0, 5),
              "C": rng.randint(0, 5), "D": rng.randint(0, 5)}
    while values["A"] == values["B"]:
        values["B"] = rng.randint(0, 5)
    def ev(cfg):
        return (cfg["A"] + cfg["B"]) * 2 + cfg["A"] * cfg["B"] + cfg["C"] * 3 + cfg["D"]
    phi = shapley_exact(values, axes, ev)
    return abs(phi["A"] - phi["B"]), values, phi


# ------------------------------------------------------------------
# minor #6: exhaustive OWAS 4-tuple neighbour enumeration
# ------------------------------------------------------------------

def exhaustive_owas_neighbours(axes, evaluator):
    """Enumerate every 4-tuple in the OWAS axis grid, then for each
    axis test +1 and -1 monotonicity. Returns per-axis violation
    counts and up to N concrete violating tuples."""
    axis_names = list(axes.keys())
    ranges = [range(lo, hi + 1) for lo, hi in axes.values()]
    per_axis_inc_violations = Counter()
    per_axis_dec_violations = Counter()
    per_axis_inc_checks = Counter()
    per_axis_dec_checks = Counter()
    examples = []
    for combo in itertools.product(*ranges):
        base_cfg = dict(zip(axis_names, combo))
        base_val = float(evaluator(base_cfg))
        for a, (lo, hi) in axes.items():
            if base_cfg[a] < hi:
                inc = dict(base_cfg); inc[a] = base_cfg[a] + 1
                per_axis_inc_checks[a] += 1
                v_inc = float(evaluator(inc))
                if v_inc < base_val - 1e-12:
                    per_axis_inc_violations[a] += 1
                    if len(examples) < 20:
                        examples.append({"kind": "increment", "axis": a,
                                         "from_cfg": dict(base_cfg), "from_v": base_val,
                                         "to_cfg": inc, "to_v": v_inc})
            if base_cfg[a] > lo:
                dec = dict(base_cfg); dec[a] = base_cfg[a] - 1
                per_axis_dec_checks[a] += 1
                v_dec = float(evaluator(dec))
                if v_dec > base_val + 1e-12:
                    per_axis_dec_violations[a] += 1
                    if len(examples) < 40:
                        examples.append({"kind": "decrement", "axis": a,
                                         "from_cfg": dict(base_cfg), "from_v": base_val,
                                         "to_cfg": dec, "to_v": v_dec})
    total_inc = sum(per_axis_inc_violations.values())
    total_dec = sum(per_axis_dec_violations.values())
    return {
        "n_tuples": functools_prod([hi - lo + 1 for lo, hi in axes.values()]),
        "per_axis_inc_violations": dict(per_axis_inc_violations),
        "per_axis_dec_violations": dict(per_axis_dec_violations),
        "per_axis_inc_checks": dict(per_axis_inc_checks),
        "per_axis_dec_checks": dict(per_axis_dec_checks),
        "total_inc_violations": total_inc,
        "total_dec_violations": total_dec,
        "total_inc_checks": sum(per_axis_inc_checks.values()),
        "total_dec_checks": sum(per_axis_dec_checks.values()),
        "example_violations": examples,
    }


def functools_prod(seq):
    p = 1
    for x in seq: p *= x
    return p


# ------------------------------------------------------------------
# minor #8: Wilson 95% CI
# ------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------
# minor #9: MC n_perm=5000 sanity subsample
# ------------------------------------------------------------------

def mc_sanity_high_budget(axes, evaluator, seeds, n_trials, low_perm, high_perm):
    """Compare max per-axis disagreement (exact vs MC) at low_perm vs
    high_perm on a small subsample. Expect approx sqrt(low/high) shrink."""
    rng_master = random.Random(sum(seeds))
    max_low, max_high = [], []
    for _ in range(n_trials):
        values = {a: rng_master.randint(lo, hi) for a, (lo, hi) in axes.items()}
        exact = shapley_exact(values, axes, evaluator)
        mc_low = shapley_perm_mc(values, axes, evaluator, low_perm, random.Random(rng_master.random()))
        mc_high = shapley_perm_mc(values, axes, evaluator, high_perm, random.Random(rng_master.random()))
        max_low.append(max(abs(exact[a] - mc_low[a]) for a in axes))
        max_high.append(max(abs(exact[a] - mc_high[a]) for a in axes))
    return {"low_perm": low_perm, "high_perm": high_perm,
            "n_trials": n_trials,
            "max_disagree_low": max(max_low),
            "max_disagree_high": max(max_high),
            "median_disagree_low": statistics.median(max_low),
            "median_disagree_high": statistics.median(max_high),
            "expected_shrink_factor_sqrt": math.sqrt(low_perm / high_perm),
            "observed_shrink_factor_median": (statistics.median(max_low) / statistics.median(max_high)) if statistics.median(max_high) > 0 else float('inf')}


# ------------------------------------------------------------------
# minor #10: dummy range/value sweep
# ------------------------------------------------------------------

def check_formal_dummy_swept(values, axes, evaluator, rng):
    """Formal dummy (N+1 axis that evaluator literally ignores) tested
    across multiple axis ranges and axis values."""
    results = []
    for lo, hi in [(1, 4), (0, 10), (-5, 5), (100, 200)]:
        for dz in [lo, (lo + hi) // 2, hi]:
            ext = dict(axes); ext["dummy_z"] = (lo, hi)
            def ev_ignore(cfg): return evaluator({k: v for k, v in cfg.items() if k != "dummy_z"})
            vals = dict(values); vals["dummy_z"] = dz
            phi = shapley_exact(vals, ext, ev_ignore)
            results.append({"range": [lo, hi], "value": dz, "shap": phi["dummy_z"]})
    return {"max_abs_shap_across_sweep": max(abs(r["shap"]) for r in results),
            "sweep": results}


# ------------------------------------------------------------------
# random-sample monotonicity (both directions, stratified)
# ------------------------------------------------------------------

def check_monotonicity_bidir_stratified(values, axes, evaluator):
    """Return {(axis, direction): {"start_bucket": ...}} shape data
    plus violation flag list."""
    base = float(evaluator(values))
    checks = []
    for a, (lo, hi) in axes.items():
        v = values[a]
        for direction, ok in (("inc", v < hi), ("dec", v > lo)):
            if not ok: continue
            v2 = dict(values); v2[a] = v + 1 if direction == "inc" else v - 1
            after = float(evaluator(v2))
            if direction == "inc":
                violated = after < base - 1e-12
            else:
                violated = after > base + 1e-12
            checks.append({"axis": a, "direction": direction,
                           "start_value": v, "delta": after - base,
                           "violated": violated})
    return checks


def run_random_mono(name, axes, evaluator, seeds, n_trials):
    """Return per-seed, per-axis, per-direction, per-start-value counts."""
    per_seed_rates = []
    total_by_axis_dir = Counter()
    viol_by_axis_dir = Counter()
    viol_by_start = Counter()
    total_by_start = Counter()
    for seed in seeds:
        rng = random.Random(seed)
        s_viol = 0; s_total = 0
        for _ in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            checks = check_monotonicity_bidir_stratified(values, axes, evaluator)
            for c in checks:
                key = (c["axis"], c["direction"])
                total_by_axis_dir[key] += 1
                total_by_start[c["start_value"]] += 1
                s_total += 1
                if c["violated"]:
                    viol_by_axis_dir[key] += 1
                    viol_by_start[c["start_value"]] += 1
                    s_viol += 1
        per_seed_rates.append(s_viol / max(s_total, 1))
    return {
        "per_seed_rate": per_seed_rates,
        "pooled_rate": sum(viol_by_axis_dir.values()) / max(sum(total_by_axis_dir.values()), 1),
        "pooled_viol": sum(viol_by_axis_dir.values()),
        "pooled_total": sum(total_by_axis_dir.values()),
        "by_axis_dir_viol": {f"{k[0]}|{k[1]}": v for k, v in viol_by_axis_dir.items()},
        "by_axis_dir_total": {f"{k[0]}|{k[1]}": v for k, v in total_by_axis_dir.items()},
        "by_start_value_viol": {str(k): v for k, v in viol_by_start.items()},
        "by_start_value_total": {str(k): v for k, v in total_by_start.items()},
    }


# ------------------------------------------------------------------
# main runners
# ------------------------------------------------------------------

def run_efficiency_and_dummy(name, axes, evaluator, seeds, n_trials):
    residuals = []
    dummy_max = []
    dummy_sweep_max = []
    for seed in seeds:
        rng = random.Random(seed)
        for corner_kind in ("min", "max"):
            v = {a: (lo if corner_kind == "min" else hi) for a, (lo, hi) in axes.items()}
            shap = shapley_exact(v, axes, evaluator)
            g = v_grand_minus_empty(v, axes, evaluator)
            residuals.append(abs(sum(shap.values()) - g))
        for _ in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            shap = shapley_exact(values, axes, evaluator)
            g = v_grand_minus_empty(values, axes, evaluator)
            residuals.append(abs(sum(shap.values()) - g))
            dsw = check_formal_dummy_swept(values, axes, evaluator, rng)
            dummy_sweep_max.append(dsw["max_abs_shap_across_sweep"])
    exact_zero_frac = sum(1 for r in residuals if r == 0.0) / len(residuals)
    return {
        "residual_max": max(residuals),
        "residual_median": statistics.median(residuals),
        "residual_exact_zero_fraction": exact_zero_frac,
        "residual_p99": statistics.quantiles(residuals, n=100)[98] if len(residuals) >= 100 else max(residuals),
        "dummy_sweep_max_shap": max(dummy_sweep_max),
    }


def run_additivity_independent(name, axes, evaluator, seeds, n_trials):
    """Additivity with an INDEPENDENT co-game (random linear over axes)."""
    max_devs = []
    for seed in seeds:
        rng = random.Random(seed)
        ev2 = make_independent_cogame(axes, seed * 101 + 7)
        for _ in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            phi1 = shapley_exact(values, axes, evaluator)
            phi2 = shapley_exact(values, axes, ev2)
            def ev_sum(cfg): return evaluator(cfg) + ev2(cfg)
            phi_sum = shapley_exact(values, axes, ev_sum)
            dev = max(abs(phi_sum[a] - (phi1[a] + phi2[a])) for a in axes)
            max_devs.append(dev)
    return {"additivity_max_dev": max(max_devs),
            "additivity_median_dev": statistics.median(max_devs)}


def run_cross_impl(name, axes, evaluator, seeds, n_trials, n_perm):
    per_seed = []
    all_max = []
    for seed in seeds:
        rng = random.Random(seed)
        s_disagrees = []
        for trial in range(n_trials):
            values = {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}
            exact = shapley_exact(values, axes, evaluator)
            per_trial_rng = random.Random(seed * 7919 + trial + 1)
            mc = shapley_perm_mc(values, axes, evaluator, n_perm, per_trial_rng)
            s_disagrees.append(max(abs(exact[a] - mc[a]) for a in axes))
        per_seed.append(max(s_disagrees))
        all_max.extend(s_disagrees)
    return {"per_seed_max_disagree": per_seed,
            "pooled_max_disagree": max(all_max),
            "pooled_median_disagree": statistics.median(all_max),
            "pooled_sd_disagree": statistics.stdev(all_max) if len(all_max) > 1 else 0.0}


def main():
    seeds = [20260808, 20260809, 20260810, 20260811, 20260812]
    n_trials = 200
    mc_perm_low = 500

    out = {"config": {"seeds": seeds, "n_trials_per_seed": n_trials,
                      "mc_permutations_low": mc_perm_low,
                      "monotonicity_denominator_notice": "excludes deterministic corners; corners contribute to efficiency/dummy stats only"}}

    schemes_out = {}
    for name, axes, ev in SCHEMES:
        s = {}
        s["efficiency_dummy"] = run_efficiency_and_dummy(name, axes, ev, seeds, n_trials)
        s["additivity_independent"] = run_additivity_independent(name, axes, ev, seeds, n_trials)
        s["cross_impl"] = run_cross_impl(name, axes, ev, seeds, n_trials, mc_perm_low)
        s["mono_random_bidir"] = run_random_mono(name, axes, ev, seeds, n_trials)
        schemes_out[name] = s
    out["schemes"] = schemes_out

    # exhaustive OWAS enumeration (small table)
    out["OWAS_exhaustive"] = exhaustive_owas_neighbours(OWAS_AXES, _eval_owas)

    # symmetry with distinct values
    sym_rng = random.Random(sum(seeds))
    sym_max = 0.0
    sym_probes = []
    for _ in range(100):
        dev, vals, phi = check_symmetry_distinct_values(sym_rng)
        sym_max = max(sym_max, dev)
        sym_probes.append({"dev": dev, "A": vals["A"], "B": vals["B"], "phi_A": phi["A"], "phi_B": phi["B"]})
    out["symmetry_distinct_values"] = {"max_dev": sym_max,
                                       "median_dev": statistics.median([p["dev"] for p in sym_probes]),
                                       "n_probes": len(sym_probes),
                                       "example_probes_first3": sym_probes[:3]}

    # MC n_perm=5000 sanity subsample (OWAS, 30 trials)
    out["mc_sanity_high_budget"] = {
        name: mc_sanity_high_budget(axes, ev, seeds[:1], 30, 500, 5000)
        for name, axes, ev in SCHEMES
    }

    # Wilson CI for OWAS random-sample violation rate
    owas_pooled = out["schemes"]["OWAS"]["mono_random_bidir"]
    inc_viol = sum(v for k, v in owas_pooled["by_axis_dir_viol"].items() if k.endswith("|inc"))
    inc_total = sum(v for k, v in owas_pooled["by_axis_dir_total"].items() if k.endswith("|inc"))
    lo_ci, hi_ci = wilson_ci(inc_viol, inc_total)
    out["owas_wilson_ci_random"] = {"inc_violations": inc_viol, "inc_checks": inc_total,
                                    "rate": inc_viol / max(inc_total, 1),
                                    "wilson_95_lo": lo_ci, "wilson_95_hi": hi_ci}

    # metrics.json (machine-readable summary)
    metrics = {
        "T16_efficiency_max_residual_owas": schemes_out["OWAS"]["efficiency_dummy"]["residual_max"],
        "T16_efficiency_max_residual_rula": schemes_out["RULA"]["efficiency_dummy"]["residual_max"],
        "T16_efficiency_max_residual_reba": schemes_out["REBA"]["efficiency_dummy"]["residual_max"],
        "T16_efficiency_exact_zero_fraction_owas": schemes_out["OWAS"]["efficiency_dummy"]["residual_exact_zero_fraction"],
        "T16_efficiency_exact_zero_fraction_rula": schemes_out["RULA"]["efficiency_dummy"]["residual_exact_zero_fraction"],
        "T16_efficiency_exact_zero_fraction_reba": schemes_out["REBA"]["efficiency_dummy"]["residual_exact_zero_fraction"],
        "T17_dummy_swept_max_shap_owas": schemes_out["OWAS"]["efficiency_dummy"]["dummy_sweep_max_shap"],
        "T17_dummy_swept_max_shap_rula": schemes_out["RULA"]["efficiency_dummy"]["dummy_sweep_max_shap"],
        "T17_dummy_swept_max_shap_reba": schemes_out["REBA"]["efficiency_dummy"]["dummy_sweep_max_shap"],
        "T18_symmetry_distinct_values_max_dev": out["symmetry_distinct_values"]["max_dev"],
        "additivity_independent_max_dev_owas": schemes_out["OWAS"]["additivity_independent"]["additivity_max_dev"],
        "additivity_independent_max_dev_rula": schemes_out["RULA"]["additivity_independent"]["additivity_max_dev"],
        "additivity_independent_max_dev_reba": schemes_out["REBA"]["additivity_independent"]["additivity_max_dev"],
        "cross_impl_per_seed_max_disagree_owas": schemes_out["OWAS"]["cross_impl"]["per_seed_max_disagree"],
        "cross_impl_sd_owas": schemes_out["OWAS"]["cross_impl"]["pooled_sd_disagree"],
        "cross_impl_sd_rula": schemes_out["RULA"]["cross_impl"]["pooled_sd_disagree"],
        "cross_impl_sd_reba": schemes_out["REBA"]["cross_impl"]["pooled_sd_disagree"],
        "mc_sanity_owas_median_disagree_500": out["mc_sanity_high_budget"]["OWAS"]["median_disagree_low"],
        "mc_sanity_owas_median_disagree_5000": out["mc_sanity_high_budget"]["OWAS"]["median_disagree_high"],
        "mc_sanity_owas_observed_shrink_factor": out["mc_sanity_high_budget"]["OWAS"]["observed_shrink_factor_median"],
        "mc_sanity_expected_shrink_factor": out["mc_sanity_high_budget"]["OWAS"]["expected_shrink_factor_sqrt"],
        # monotonicity: exhaustive OWAS + random-sample stratified
        "monotonicity_owas_exhaustive_inc_violations": out["OWAS_exhaustive"]["total_inc_violations"],
        "monotonicity_owas_exhaustive_inc_checks": out["OWAS_exhaustive"]["total_inc_checks"],
        "monotonicity_owas_exhaustive_inc_rate": out["OWAS_exhaustive"]["total_inc_violations"] / max(out["OWAS_exhaustive"]["total_inc_checks"], 1),
        "monotonicity_owas_exhaustive_dec_violations": out["OWAS_exhaustive"]["total_dec_violations"],
        "monotonicity_owas_exhaustive_dec_checks": out["OWAS_exhaustive"]["total_dec_checks"],
        "monotonicity_owas_exhaustive_dec_rate": out["OWAS_exhaustive"]["total_dec_violations"] / max(out["OWAS_exhaustive"]["total_dec_checks"], 1),
        "monotonicity_owas_per_axis_inc_violations": out["OWAS_exhaustive"]["per_axis_inc_violations"],
        "monotonicity_owas_per_axis_inc_checks": out["OWAS_exhaustive"]["per_axis_inc_checks"],
        "monotonicity_owas_per_axis_dec_violations": out["OWAS_exhaustive"]["per_axis_dec_violations"],
        "monotonicity_owas_per_axis_dec_checks": out["OWAS_exhaustive"]["per_axis_dec_checks"],
        "monotonicity_random_pooled_owas": schemes_out["OWAS"]["mono_random_bidir"]["pooled_rate"],
        "monotonicity_random_pooled_rula": schemes_out["RULA"]["mono_random_bidir"]["pooled_rate"],
        "monotonicity_random_pooled_reba": schemes_out["REBA"]["mono_random_bidir"]["pooled_rate"],
        "monotonicity_random_checks_owas": schemes_out["OWAS"]["mono_random_bidir"]["pooled_total"],
        "monotonicity_random_checks_rula": schemes_out["RULA"]["mono_random_bidir"]["pooled_total"],
        "monotonicity_random_checks_reba": schemes_out["REBA"]["mono_random_bidir"]["pooled_total"],
        "monotonicity_random_per_seed_owas": schemes_out["OWAS"]["mono_random_bidir"]["per_seed_rate"],
        "owas_wilson_95_lo": out["owas_wilson_ci_random"]["wilson_95_lo"],
        "owas_wilson_95_hi": out["owas_wilson_ci_random"]["wilson_95_hi"],
        "seeds": seeds,
    }

    result_dir = Path(os.environ.get("PWW_RESULT_DIR", TOPIC_ROOT / "result" / "exp-001c-shapley-hardening"))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (result_dir / "raw.json").write_text(json.dumps(out, indent=2, default=str))

    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list) or len(v) < 10}, indent=2))


if __name__ == "__main__":
    main()
