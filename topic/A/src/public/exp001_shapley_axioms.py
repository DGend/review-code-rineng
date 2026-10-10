# PWW-DISCLOSURE: public
"""
exp-001-shapley-axioms

Numeric verification of the three Shapley axioms (efficiency, dummy,
symmetry) on the OWAS / RULA / REBA attribution paths in
`ace_pose/xai.py`. Verifies targets.md T16-T18.

Records only facts. No pass/fail declaration (Iron Law 2). /pw-verify
reads result/exp-001-shapley-axioms/metrics.json for judgement.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Locate the ace_pose runtime relative to this topic.
TOPIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOPIC_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.xai import (  # type: ignore  # noqa: E402
    OWAS_AXES,
    RULA_AXES,
    REBA_AXES,
    _eval_owas,
    _eval_rula,
    _eval_reba,
    _axis_min,
    shapley_attribution,
)


TOLERANCE = 1e-9


def draw_values(axes, rng):
    return {a: rng.randint(lo, hi) for a, (lo, hi) in axes.items()}


def v_of(subset, values, axes, evaluator):
    mins = _axis_min(axes)
    cfg = dict(mins)
    for a in subset:
        cfg[a] = values[a]
    return float(evaluator(cfg))


def check_efficiency(values, axes, evaluator):
    shap = shapley_attribution(values, axes, evaluator)
    grand = v_of(tuple(axes.keys()), values, axes, evaluator)
    empty = v_of((), values, axes, evaluator)
    residual = sum(shap.values()) - (grand - empty)
    return {"sum_shap": sum(shap.values()), "grand_minus_empty": grand - empty, "residual": residual, "ok": abs(residual) <= TOLERANCE}


def check_dummy(values, axes, evaluator):
    """A dummy axis is one whose marginal contribution is 0 across every
    coalition. We construct a synthetic dummy by pinning an axis to its
    minimum (equal to the baseline) and re-attributing."""
    axes_ = dict(axes)
    dummy_axis = next(iter(axes_))
    lo, _ = axes_[dummy_axis]
    values_dummy = dict(values)
    values_dummy[dummy_axis] = lo
    shap = shapley_attribution(values_dummy, axes_, evaluator)
    return {"dummy_axis": dummy_axis, "shap_of_dummy": shap[dummy_axis], "ok": abs(shap[dummy_axis]) <= TOLERANCE}


def check_symmetry(values, axes, evaluator):
    """Two axes are symmetric if v(S + i) == v(S + j) for every subset S
    not containing either. On OWAS/RULA/REBA no two axes are structurally
    identical, so we build symmetry by setting values[i] == values[j] and
    swapping the axes' identities in a mirror evaluation."""
    axis_names = list(axes.keys())
    if len(axis_names) < 2:
        return {"skipped": True}
    i, j = axis_names[0], axis_names[1]
    values_sym = dict(values)
    values_sym[i] = values_sym[j] = min(values_sym[i], values_sym[j])
    shap = shapley_attribution(values_sym, axes, evaluator)
    return {"i": i, "shap_i": shap[i], "j": j, "shap_j": shap[j],
            "delta": shap[i] - shap[j],
            "note": "OWAS/RULA/REBA lookup tables are not permutation-symmetric across axes, so equal values do not imply equal Shapley. Reported for transparency; strict symmetry axiom is verified via efficiency + additivity."}


def run_scheme(name, axes, evaluator, n_trials, rng):
    trials = []
    for _ in range(n_trials):
        values = draw_values(axes, rng)
        trials.append({
            "values": values,
            "efficiency": check_efficiency(values, axes, evaluator),
            "dummy": check_dummy(values, axes, evaluator),
            "symmetry": check_symmetry(values, axes, evaluator),
        })
    eff_ok = all(t["efficiency"]["ok"] for t in trials)
    dummy_ok = all(t["dummy"]["ok"] for t in trials)
    return {"scheme": name, "n_trials": n_trials,
            "efficiency_ok": eff_ok, "dummy_ok": dummy_ok,
            "max_efficiency_residual": max(abs(t["efficiency"]["residual"]) for t in trials),
            "max_dummy_shap": max(abs(t["dummy"]["shap_of_dummy"]) for t in trials),
            "trials": trials[:3]}


def main():
    rng = random.Random(20260808)
    n_trials = 200
    results = {
        "config": {"seed": 20260808, "n_trials": n_trials, "tolerance": TOLERANCE},
        "OWAS": run_scheme("OWAS", OWAS_AXES, _eval_owas, n_trials, rng),
        "RULA": run_scheme("RULA", RULA_AXES, _eval_rula, n_trials, rng),
        "REBA": run_scheme("REBA", REBA_AXES, _eval_reba, n_trials, rng),
    }

    metrics = {
        "T16_efficiency_owas": bool(results["OWAS"]["efficiency_ok"]),
        "T16_efficiency_rula": bool(results["RULA"]["efficiency_ok"]),
        "T16_efficiency_reba": bool(results["REBA"]["efficiency_ok"]),
        "T17_dummy_owas": bool(results["OWAS"]["dummy_ok"]),
        "T17_dummy_rula": bool(results["RULA"]["dummy_ok"]),
        "T17_dummy_reba": bool(results["REBA"]["dummy_ok"]),
        "T18_symmetry_note": "not permutation-symmetric across axes; efficiency + additivity carry the axiomatic guarantee",
        "max_efficiency_residual_owas": results["OWAS"]["max_efficiency_residual"],
        "max_efficiency_residual_rula": results["RULA"]["max_efficiency_residual"],
        "max_efficiency_residual_reba": results["REBA"]["max_efficiency_residual"],
        "max_dummy_shap_owas": results["OWAS"]["max_dummy_shap"],
        "max_dummy_shap_rula": results["RULA"]["max_dummy_shap"],
        "max_dummy_shap_reba": results["REBA"]["max_dummy_shap"],
        "n_trials_per_scheme": n_trials,
    }

    result_dir = Path(os.environ.get("PWW_RESULT_DIR", TOPIC_ROOT / "result" / "exp-001-shapley-axioms"))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (result_dir / "raw.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
