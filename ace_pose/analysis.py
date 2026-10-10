"""Modular statistical analysis for ace_pose batch results.

Given a DataFrame produced by `ace_pose.batch_infer` (with columns like
`owas_action`, `rula_grand`, `reba_grand`, `domain`, `factory_id`,
`process_label`, `kpt_conf_mean`, …), this module produces:

  - per-group summaries (mean / median / distribution per method)
  - cross-method agreement (Cohen's κ, weighted κ, Spearman ρ)
  - between-group statistical tests (chi-square, Kruskal-Wallis)
  - confidence-stratified analyses (high- vs low-confidence pose subsets)
  - top-risk groups by any user-supplied key

All outputs are pandas DataFrames or dicts — no matplotlib dependency
at the analysis layer (plotting belongs to a separate reporting layer
that can import this module's results).

Re-usable across pipeline versions: when the Y module adds a
`pred_process_id` column, calling `analyzer.by(["pred_process_id"])`
yields the same statistics on the new grouping.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---- canonical column names produced by batch_infer ------------------

METHOD_COLS = {
    "owas": {"score": "owas_action", "range": (1, 4)},
    "rula": {"score": "rula_grand", "range": (1, 7)},
    "reba": {"score": "reba_grand", "range": (1, 15)},
}

CONFIDENCE_COL = "kpt_conf_mean"

# Optional friendly labels for the seven factories per the project PDF.
FACTORY_LABEL = {
    1: "F1-DFI",
    2: "F2-Wonkwang",
    3: "F3-SUNTECH",    # SUNTECH
    4: "F4-Joyang",
    5: "F5-Kumkangseong",
    6: "F6-Eldercare",
    7: "F7-Smile-Move",
}

# Domain rollups (preferred terminal categorization).
DOMAIN_LABEL = {
    "manufacturing": "Manufacturing",
    "eldercare": "Eldercare",
    "logistics": "Logistics",
    "unknown": "Unknown",
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def cohen_kappa(a, b, weights: str = "linear") -> float | None:
    """Weighted Cohen's κ between two integer score arrays.

    `weights` ∈ {"linear", "quadratic", None}. Returns None when scores
    are constant (κ undefined).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, weights=weights))


def spearman_rho(a, b) -> tuple[float, float] | tuple[None, None]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 3:
        return None, None
    rho, p = stats.spearmanr(a, b)
    if np.isnan(rho):
        return None, None
    return float(rho), float(p)


def kruskal_wallis(df: pd.DataFrame, score_col: str, group_col: str) -> dict:
    """One-way Kruskal-Wallis test of `score_col` across groups in `group_col`.

    Returns dict with H, p, n_groups, n_total, group sizes.
    """
    groups = []
    keys = []
    for k, sub in df.groupby(group_col):
        v = sub[score_col].dropna().to_numpy()
        if v.size >= 2:
            groups.append(v)
            keys.append(k)
    if len(groups) < 2:
        return {"H": None, "p": None, "n_groups": len(groups), "groups": keys}
    H, p = stats.kruskal(*groups)
    return {
        "H": float(H), "p": float(p), "n_groups": len(groups),
        "n_total": sum(g.size for g in groups), "groups": keys,
    }


def chi_square_distribution(df: pd.DataFrame, score_col: str, group_col: str) -> dict:
    """Chi-square test on the contingency table of `score_col` × `group_col`."""
    ct = pd.crosstab(df[group_col], df[score_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": None, "p": None, "dof": 0, "n": int(ct.values.sum())}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(ct.values.sum())}


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------

@dataclass
class GroupSummary:
    by: list[str]
    n: pd.DataFrame                           # group counts
    score_stats: dict[str, pd.DataFrame]      # method → mean/median/std DataFrame
    action_dist: dict[str, pd.DataFrame]      # method → action-level counts pivot
    confidence: pd.DataFrame                  # confidence stats per group

    def to_dict(self) -> dict:
        return {
            "by": self.by,
            "n": self.n.reset_index().to_dict(orient="records"),
            "score_stats": {k: v.reset_index().to_dict(orient="records")
                            for k, v in self.score_stats.items()},
            "action_dist": {k: v.reset_index().to_dict(orient="records")
                            for k, v in self.action_dist.items()},
            "confidence": self.confidence.reset_index().to_dict(orient="records"),
        }


@dataclass
class AgreementReport:
    """Inter-method agreement metrics (one row per method pair)."""
    pairs: pd.DataFrame   # columns: method_a, method_b, kappa_linear, kappa_quadratic, rho, n


@dataclass
class GroupDifferenceReport:
    """Statistical tests for between-group differences in score."""
    method: str
    group_col: str
    chi2: dict
    kw: dict


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------

@dataclass
class ErgoAnalyzer:
    df: pd.DataFrame
    method_cols: dict[str, dict] = field(default_factory=lambda: dict(METHOD_COLS))

    def __post_init__(self):
        # Filter to only successful rows
        if "status" in self.df.columns:
            self.df = self.df.loc[self.df["status"] == "ok"].copy()

    # -------- group summaries --------

    def by(self, group_cols: list[str]) -> GroupSummary:
        if not group_cols:
            raise ValueError("group_cols must be non-empty")
        for c in group_cols:
            if c not in self.df.columns:
                raise KeyError(f"missing column: {c}")

        n = self.df.groupby(group_cols).size().rename("n").to_frame()

        score_stats: dict[str, pd.DataFrame] = {}
        action_dist: dict[str, pd.DataFrame] = {}
        for method, cfg in self.method_cols.items():
            col = cfg["score"]
            if col not in self.df.columns:
                continue
            grp = self.df.groupby(group_cols)[col]
            score_stats[method] = grp.agg(["count", "mean", "median", "std", "min", "max"])
            ct = (self.df.groupby(group_cols + [col]).size()
                  .unstack(fill_value=0))
            action_dist[method] = ct

        conf_col = CONFIDENCE_COL if CONFIDENCE_COL in self.df.columns else None
        if conf_col is not None:
            confidence = (self.df.groupby(group_cols)[conf_col]
                          .agg(["count", "mean", "median", "std", "min", "max"]))
        else:
            confidence = pd.DataFrame()
        return GroupSummary(
            by=group_cols, n=n,
            score_stats=score_stats, action_dist=action_dist,
            confidence=confidence,
        )

    # -------- cross-method agreement --------

    def cross_method_agreement(self) -> AgreementReport:
        rows = []
        methods = [m for m, cfg in self.method_cols.items() if cfg["score"] in self.df.columns]
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                a = self.df[self.method_cols[ma]["score"]].dropna()
                b = self.df[self.method_cols[mb]["score"]].dropna()
                idx = a.index.intersection(b.index)
                a, b = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
                rho, _p = spearman_rho(a, b)
                rows.append({
                    "method_a": ma, "method_b": mb,
                    "kappa_linear": cohen_kappa(a, b, weights="linear"),
                    "kappa_quadratic": cohen_kappa(a, b, weights="quadratic"),
                    "rho": rho, "n": int(a.size),
                })
        return AgreementReport(pairs=pd.DataFrame(rows))

    # -------- group-difference tests --------

    def group_difference(self, method: str, group_col: str) -> GroupDifferenceReport:
        col = self.method_cols[method]["score"]
        chi = chi_square_distribution(self.df, col, group_col)
        kw = kruskal_wallis(self.df, col, group_col)
        return GroupDifferenceReport(
            method=method, group_col=group_col, chi2=chi, kw=kw,
        )

    # -------- confidence-stratified --------

    def confidence_strata(
        self, low: float = 0.5, high: float = 0.7,
    ) -> dict[str, GroupSummary]:
        """Split the data into low/mid/high confidence strata and run
        per-stratum analysis."""
        if CONFIDENCE_COL not in self.df.columns:
            raise KeyError(CONFIDENCE_COL)
        df = self.df.copy()
        df["conf_band"] = pd.cut(
            df[CONFIDENCE_COL], bins=[-np.inf, low, high, np.inf],
            labels=["low", "mid", "high"],
        )
        out = {}
        for band, sub in df.groupby("conf_band", observed=False):
            if sub.empty:
                continue
            out[str(band)] = ErgoAnalyzer(sub).by(["domain"])
        return out

    # -------- top groups --------

    def top_groups(
        self, method: str, group_col: str,
        agg: str = "mean", min_n: int = 20, k: int = 10,
    ) -> pd.DataFrame:
        col = self.method_cols[method]["score"]
        g = self.df.groupby(group_col)[col].agg(["count", agg])
        g = g.loc[g["count"] >= min_n]
        return g.sort_values(agg, ascending=False).head(k)


# ----------------------------------------------------------------------
# Report writer — saves an entire analysis to a directory
# ----------------------------------------------------------------------

def export_report(analyzer: ErgoAnalyzer, out_dir: Path,
                  group_specs: list[list[str]] | None = None) -> Path:
    """Write a multi-CSV bundle: one for each group breakdown plus the
    cross-method agreement and per-domain Kruskal/chi-square tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_specs is None:
        group_specs = [["domain"], ["factory_id"], ["process_label"]]

    for cols in group_specs:
        try:
            s = analyzer.by(cols)
        except KeyError as ex:
            warnings.warn(f"skip group {cols}: {ex}")
            continue
        tag = "_".join(cols)
        s.n.to_csv(out_dir / f"n_by_{tag}.csv")
        for m, df in s.score_stats.items():
            df.to_csv(out_dir / f"{m}_stats_by_{tag}.csv")
        for m, df in s.action_dist.items():
            df.to_csv(out_dir / f"{m}_actiondist_by_{tag}.csv")
        if not s.confidence.empty:
            s.confidence.to_csv(out_dir / f"confidence_by_{tag}.csv")

    agr = analyzer.cross_method_agreement()
    agr.pairs.to_csv(out_dir / "cross_method_agreement.csv", index=False)

    # group-difference tests by domain
    test_rows = []
    if "domain" in analyzer.df.columns:
        for m in analyzer.method_cols:
            if analyzer.method_cols[m]["score"] in analyzer.df.columns:
                gd = analyzer.group_difference(m, "domain")
                test_rows.append({
                    "method": m, "group_col": "domain",
                    "chi2": gd.chi2.get("chi2"), "chi2_p": gd.chi2.get("p"),
                    "chi2_dof": gd.chi2.get("dof"),
                    "kw_H": gd.kw.get("H"), "kw_p": gd.kw.get("p"),
                    "kw_n_groups": gd.kw.get("n_groups"),
                })
    pd.DataFrame(test_rows).to_csv(out_dir / "group_difference_domain.csv", index=False)

    return out_dir
