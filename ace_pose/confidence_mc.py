"""C3 — Confidence-aware probabilistic ergonomic scoring via Monte Carlo.

Standard OWAS/RULA/REBA scoring thresholds joint angles into discrete
categorical risk classes. A small keypoint estimation error near a
threshold (e.g., neck flexion 19° vs 21° with a cutoff at 20°) causes a
large risk-class flip.

RTMW3D outputs per-keypoint confidence c_i ∈ [0, 1]. This module:
  1. Models each keypoint as a Gaussian with σ_i = σ_max · (1 - c_i),
     where σ_max is calibrated as a fraction of the body's reference
     length (so it is anthropometry-invariant).
  2. Draws N samples of the pose, runs the full ergo_v2 pipeline on
     each, and aggregates into a per-class probability distribution.
  3. Optionally applies temperature scaling to the resulting distribution
     to minimise NLL on a held-out validation set.

Outputs reliability metrics (Brier score, ECE) when ground-truth labels
are supplied.

Gap addressed (see `standards/audit.md §B.2 and §C.2`): legacy systems
report a single integer; experts disagree at thresholds; downstream
intervention decisions are coarser than the underlying angle data
supports. This module turns the score into a calibrated distribution
suitable for clinical / safety decision-making.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ace_pose.major_joints import Major25Pose
from ace_pose.anthropometry import compute_anthropometry
from ace_pose.ergo_v2 import score_all
from ace_pose.ergo_v2.result import ErgoResult


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# σ_max as a fraction of `ref_length` (max of shoulder/hip/torso). The
# default 0.04 is calibrated to match the empirical 95 % keypoint-error
# quantile we observe on RTMW3D-x at our 6,007-image domain (validated
# separately during C3 calibration).
DEFAULT_SIGMA_FRACTION = 0.04


@dataclass
class McConfig:
    n_samples: int = 256
    sigma_fraction: float = DEFAULT_SIGMA_FRACTION
    rng_seed: int | None = 0


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def sample_poses(
    pose: Major25Pose, config: McConfig,
) -> Sequence[Major25Pose]:
    """Draw N noisy realisations of the pose with per-keypoint σ ∝ (1-conf).

    Returns a list of `Major25Pose` instances sharing the same `source`
    field (copied) but with perturbed `keypoints`.
    """
    rng = np.random.default_rng(config.rng_seed)
    anthro = compute_anthropometry(pose)
    sigma_base = config.sigma_fraction * max(anthro.ref_length, 1e-6)

    # σ per keypoint: high-confidence → small noise, low-confidence → large
    sigma_per_joint = sigma_base * (1.0 - np.clip(pose.confidence, 0.0, 1.0))
    sigma_per_joint = sigma_per_joint[:, None]   # (25, 1)

    out: list[Major25Pose] = []
    for _ in range(config.n_samples):
        noise = rng.normal(0.0, 1.0, size=pose.keypoints.shape) * sigma_per_joint
        sample_kp = pose.keypoints + noise
        out.append(Major25Pose(
            keypoints=sample_kp,
            confidence=pose.confidence,
            source=pose.source,
        ))
    return out


# ----------------------------------------------------------------------
# Distributional scoring
# ----------------------------------------------------------------------

@dataclass
class RiskDistribution:
    """Probability distribution over an ergo method's discrete risk classes."""
    method: str                      # "owas" / "rula" / "reba"
    classes: np.ndarray              # 1-D int array of class labels
    probs: np.ndarray                # matching probabilities, sum to 1
    n_samples: int

    @property
    def mode(self) -> int:
        return int(self.classes[np.argmax(self.probs)])

    @property
    def expected(self) -> float:
        return float(np.dot(self.classes, self.probs))

    @property
    def variance(self) -> float:
        e = self.expected
        return float(np.dot(self.probs, (self.classes - e) ** 2))

    @property
    def entropy(self) -> float:
        p = self.probs + 1e-12
        return float(-(p * np.log(p)).sum())

    def class_probability(self, cls: int) -> float:
        idx = np.where(self.classes == cls)[0]
        return float(self.probs[idx[0]]) if idx.size else 0.0


def _aggregate(samples: Sequence[int], min_cls: int, max_cls: int) -> RiskDistribution:
    classes = np.arange(min_cls, max_cls + 1, dtype=int)
    counts = np.zeros(classes.shape[0], dtype=np.int64)
    for s in samples:
        if min_cls <= s <= max_cls:
            counts[s - min_cls] += 1
    probs = counts / max(counts.sum(), 1)
    return RiskDistribution(
        method="", classes=classes, probs=probs, n_samples=len(samples),
    )


@dataclass
class McErgoResult:
    """All three risk distributions plus the deterministic baseline result
    for traceability."""
    owas_dist: RiskDistribution
    rula_dist: RiskDistribution
    reba_dist: RiskDistribution
    baseline: ErgoResult


def mc_score_all(
    pose: Major25Pose,
    *,
    config: McConfig | None = None,
    world_up: np.ndarray | None = None,
    load_kg: float = 0.0,
    walking: bool = False,
    coupling: int = 1,
    # Must match the deterministic cohort scorer (score_all / batch_infer), whose
    # scores are the calibration labels; a True default here shifted every
    # sampled RULA/REBA score by the activity add-on (fixed 2026-10-05).
    static_minute: bool = False,
    repetitive_4_per_min: bool = False,
    load_repeated: bool = False,
    rapid_changes: bool = False,
    process_id: int | None = None,
    domain: str | None = None,
) -> McErgoResult:
    """Score one pose with confidence-aware Monte Carlo over all three methods.

    Returns:
        McErgoResult with one `RiskDistribution` per method plus the
        deterministic ergo_v2 result on the un-perturbed pose for direct
        comparison.
    """
    if config is None:
        config = McConfig()

    # deterministic baseline on un-perturbed pose
    baseline = score_all(
        pose, world_up=world_up, load_kg=load_kg, walking=walking,
        coupling=coupling,
        static_minute=static_minute,
        repetitive_4_per_min=repetitive_4_per_min,
        load_repeated=load_repeated,
        rapid_changes=rapid_changes,
        process_id=process_id, domain=domain,
        bilateral=True,
    )

    owas_samples: list[int] = []
    rula_samples: list[int] = []
    reba_samples: list[int] = []

    for sampled in sample_poses(pose, config):
        try:
            r = score_all(
                sampled, world_up=world_up, load_kg=load_kg, walking=walking,
                coupling=coupling,
                static_minute=static_minute,
                repetitive_4_per_min=repetitive_4_per_min,
        load_repeated=load_repeated,
                rapid_changes=rapid_changes,
                process_id=process_id, domain=domain,
                bilateral=True,
            )
            owas_samples.append(r.owas.action_category)
            rula_samples.append(r.rula.grand_score)
            reba_samples.append(r.reba.grand_score)
        except Exception:
            # robust: skip degenerate MC samples
            continue

    owas_dist = _aggregate(owas_samples, 1, 4)
    owas_dist.method = "owas"
    rula_dist = _aggregate(rula_samples, 1, 7)
    rula_dist.method = "rula"
    reba_dist = _aggregate(reba_samples, 1, 15)
    reba_dist.method = "reba"

    return McErgoResult(
        owas_dist=owas_dist, rula_dist=rula_dist, reba_dist=reba_dist,
        baseline=baseline,
    )


# ----------------------------------------------------------------------
# Temperature scaling and calibration metrics
# ----------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Standard temperature scaling: p_T = softmax(log p / T)."""
    log_p = np.log(np.clip(probs, 1e-12, 1.0)) / max(T, 1e-6)
    log_p = log_p - log_p.max()
    e = np.exp(log_p)
    return e / e.sum()


def fit_temperature(
    prob_dists: Sequence[RiskDistribution],
    true_labels: Sequence[int],
    grid: np.ndarray | None = None,
) -> float:
    """Pick T ∈ grid that minimises NLL across the given samples.

    Each `prob_dists[i]` must be a distribution over `prob_dists[i].classes`
    and `true_labels[i]` must be one of those classes.
    """
    if grid is None:
        grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    if len(prob_dists) != len(true_labels):
        raise ValueError("prob_dists and true_labels length mismatch")
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = 0.0
        for dist, y in zip(prob_dists, true_labels):
            p_cal = apply_temperature(dist.probs, T)
            idx = np.where(dist.classes == y)[0]
            if idx.size == 0:
                continue
            nll -= np.log(p_cal[idx[0]] + 1e-12)
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def brier_score(prob_dists: Sequence[RiskDistribution], true_labels: Sequence[int]) -> float:
    """Multi-class Brier score: 1/N · Σ_i Σ_c (p_{i,c} − I[y_i = c])²."""
    total = 0.0
    n = 0
    for dist, y in zip(prob_dists, true_labels):
        one_hot = (dist.classes == y).astype(float)
        total += float(np.sum((dist.probs - one_hot) ** 2))
        n += 1
    return total / max(n, 1)


def expected_calibration_error(
    prob_dists: Sequence[RiskDistribution], true_labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) using top-class confidence."""
    if not prob_dists:
        return 0.0
    confs, corrects = [], []
    for dist, y in zip(prob_dists, true_labels):
        c = float(np.max(dist.probs))
        pred = int(dist.classes[np.argmax(dist.probs)])
        confs.append(c)
        corrects.append(1.0 if pred == int(y) else 0.0)
    confs = np.array(confs)
    corrects = np.array(corrects)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = confs.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs > lo) & (confs <= hi) if hi < 1.0 else (confs > lo) & (confs <= hi + 1e-9)
        if mask.sum() == 0:
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)
