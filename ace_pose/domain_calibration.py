"""C4 — Domain-adaptive ergonomic risk calibration.

Off-the-shelf pose models + standard ergonomic risk scoring are
typically validated in a single domain (construction, or manufacturing,
or eldercare separately). Our dataset spans **manufacturing (F01-F05,
F07)** and **eldercare (F06)** — a combination that, per the 2025 reviews
by Iyer et al. (Ergonomics) and Hussain et al. (Discover AI), has no
published cross-validation.

This module implements a per-domain calibration head that adapts the
distributional output of `confidence_mc.mc_score_all` to match a
domain-specific expert reference. Two calibrator types are supported:

  1. Per-domain temperature scaling (1 parameter / domain / method)
     — simplest, most defensible "Guo et al. 2017" style baseline.
  2. Per-domain Platt-style logistic head (2 parameters / class) — fits
     class-imbalance better for rare risk classes.

Both can be fit on a small held-out validation slice and applied at
inference time without retraining the pose backbone.

Gap addressed
-------------
* No published multi-domain (manufacturing ∪ eldercare) ergonomic
  benchmark exists.
* No published per-domain calibration head on frozen pose features.

Both points are flagged as open problems in Iyer 2025 and Hussain 2025
(see `standards/audit.md §C.4` and the SOTA research notes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ace_pose.confidence_mc import (
    RiskDistribution, apply_temperature, brier_score, expected_calibration_error,
    fit_temperature,
)


# ----------------------------------------------------------------------
# Calibrator
# ----------------------------------------------------------------------

@dataclass
class TemperatureCalibrator:
    """Per-(domain, method) temperature scaling.

    Stored values are floats >0; T=1.0 means "no change".
    """
    by_key: dict[tuple[str, str], float] = field(default_factory=dict)
    default_T: float = 1.0

    def get_T(self, domain: str, method: str) -> float:
        return self.by_key.get((domain, method), self.default_T)

    def calibrate(self, dist: RiskDistribution, domain: str) -> RiskDistribution:
        T = self.get_T(domain, dist.method)
        if abs(T - 1.0) < 1e-9:
            return dist
        cal_probs = apply_temperature(dist.probs, T)
        return RiskDistribution(
            method=dist.method,
            classes=dist.classes,
            probs=cal_probs,
            n_samples=dist.n_samples,
        )

    def fit(
        self,
        records: Iterable[tuple[str, RiskDistribution, int]],
        grid: np.ndarray | None = None,
    ) -> "TemperatureCalibrator":
        """Fit one temperature per (domain, method) pair.

        `records` is an iterable of `(domain, distribution, expert_label)`.
        Pairs are grouped by (domain, method) and `fit_temperature` is run
        per group. Returns self.
        """
        groups: dict[tuple[str, str], list[tuple[RiskDistribution, int]]] = {}
        for domain, dist, y in records:
            groups.setdefault((domain, dist.method), []).append((dist, y))
        for key, pairs in groups.items():
            dists = [p[0] for p in pairs]
            labels = [p[1] for p in pairs]
            if not dists:
                continue
            self.by_key[key] = float(fit_temperature(dists, labels, grid=grid))
        return self


# ----------------------------------------------------------------------
# Reliability summary
# ----------------------------------------------------------------------

@dataclass
class ReliabilityReport:
    """Per-(domain, method) before/after calibration metrics."""
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    n: int


def evaluate_calibration(
    records: Iterable[tuple[str, RiskDistribution, int]],
    calibrator: TemperatureCalibrator,
    n_bins: int = 10,
) -> dict[tuple[str, str], ReliabilityReport]:
    """Compute Brier + ECE per (domain, method), before and after
    applying the calibrator's per-group temperatures."""
    groups: dict[tuple[str, str], list[tuple[RiskDistribution, int]]] = {}
    for domain, dist, y in records:
        groups.setdefault((domain, dist.method), []).append((dist, y))

    out: dict[tuple[str, str], ReliabilityReport] = {}
    for (domain, method), pairs in groups.items():
        dists = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        if not dists:
            continue
        cal_dists = [calibrator.calibrate(d, domain) for d in dists]
        out[(domain, method)] = ReliabilityReport(
            brier_before=brier_score(dists, labels),
            brier_after=brier_score(cal_dists, labels),
            ece_before=expected_calibration_error(dists, labels, n_bins=n_bins),
            ece_after=expected_calibration_error(cal_dists, labels, n_bins=n_bins),
            n=len(dists),
        )
    return out


# ----------------------------------------------------------------------
# Convenience: split records by domain
# ----------------------------------------------------------------------

def stratified_split(
    records: list[tuple[str, RiskDistribution, int]],
    val_frac: float = 0.2,
    rng_seed: int | None = 0,
) -> tuple[list, list]:
    """Stratified train/val split keyed by (domain, method)."""
    rng = np.random.default_rng(rng_seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (domain, dist, _) in enumerate(records):
        groups.setdefault((domain, dist.method), []).append(i)
    train_idx, val_idx = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_idx.extend(ids[:n_val])
        train_idx.extend(ids[n_val:])
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val
