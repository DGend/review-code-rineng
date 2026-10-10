"""Tests for `ace_pose.analysis`."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.analysis import (
    AgreementReport, ErgoAnalyzer, FACTORY_LABEL, GroupSummary,
    chi_square_distribution, cohen_kappa, export_report, kruskal_wallis,
    spearman_rho,
)


def _synthetic_batch_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    domains = ["manufacturing", "eldercare", "logistics"]
    factories = [1, 2, 3, 4, 5, 6, 7]
    processes = ["사상1", "용접1", "주조", "조립", "이동", "마사지"]
    for i in range(n):
        d = rng.choice(domains)
        if d == "manufacturing":
            f = int(rng.choice([1, 2, 3, 4, 5]))
            rula_mean = 3.5
        elif d == "eldercare":
            f = 6
            rula_mean = 2.5
        else:
            f = 7
            rula_mean = 2.8
        rows.append({
            "status": "ok",
            "domain": d, "factory_id": f, "process_label": rng.choice(processes),
            "owas_action": int(rng.integers(1, 5)),
            "rula_grand": int(np.clip(rng.normal(rula_mean, 1.0), 1, 7)),
            "reba_grand": int(np.clip(rng.normal(rula_mean + 2, 1.5), 1, 15)),
            "kpt_conf_mean": float(rng.uniform(0.35, 0.85)),
        })
    return pd.DataFrame(rows)


class TestHelpers:
    def test_cohen_kappa_constant_returns_none(self):
        a = [1, 1, 1, 1]
        b = [1, 1, 1, 1]
        assert cohen_kappa(a, b) is None

    def test_cohen_kappa_perfect_agreement(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        assert cohen_kappa(a, b) == pytest.approx(1.0)

    def test_spearman_perfect(self):
        rho, p = spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert rho == pytest.approx(1.0)
        assert p < 0.05

    def test_chi_square_rejects_when_distributions_differ(self):
        df = pd.DataFrame({
            "g": ["a"] * 50 + ["b"] * 50,
            "score": [1] * 50 + [3] * 50,
        })
        r = chi_square_distribution(df, "score", "g")
        assert r["p"] < 0.05

    def test_kruskal_wallis_rejects_when_distributions_differ(self):
        df = pd.DataFrame({
            "g": ["a"] * 50 + ["b"] * 50,
            "score": list(range(50)) + list(range(100, 150)),
        })
        r = kruskal_wallis(df, "score", "g")
        assert r["p"] < 0.05


class TestAnalyzer:
    def test_filters_failed_rows(self):
        df = _synthetic_batch_df(40)
        df.loc[0:5, "status"] = "no_detection"
        a = ErgoAnalyzer(df)
        assert (a.df["status"] == "ok").all()
        assert len(a.df) == len(df) - 6

    def test_by_domain_summary(self):
        df = _synthetic_batch_df(150)
        a = ErgoAnalyzer(df)
        s = a.by(["domain"])
        assert isinstance(s, GroupSummary)
        assert "owas" in s.score_stats and "rula" in s.score_stats
        assert "owas" in s.action_dist
        # n should sum to number of rows
        assert int(s.n["n"].sum()) == len(df)

    def test_by_factory_summary(self):
        df = _synthetic_batch_df(200, seed=2)
        a = ErgoAnalyzer(df)
        s = a.by(["factory_id"])
        assert s.n.shape[0] >= 3
        # confidence summary present
        assert not s.confidence.empty

    def test_cross_method_agreement(self):
        df = _synthetic_batch_df(200, seed=3)
        a = ErgoAnalyzer(df)
        rep = a.cross_method_agreement()
        assert isinstance(rep, AgreementReport)
        # We have 3 methods → 3 pairs
        assert len(rep.pairs) == 3

    def test_group_difference_significant_on_synthetic(self):
        df = _synthetic_batch_df(400, seed=4)
        a = ErgoAnalyzer(df)
        gd = a.group_difference("rula", "domain")
        # In synthetic data, manufacturing has higher RULA mean →
        # tests should be significant
        assert gd.kw["p"] is None or gd.kw["p"] < 0.05

    def test_confidence_strata_splits(self):
        df = _synthetic_batch_df(300, seed=5)
        a = ErgoAnalyzer(df)
        strata = a.confidence_strata(low=0.5, high=0.7)
        assert len(strata) >= 1
        for band, summ in strata.items():
            assert isinstance(summ, GroupSummary)

    def test_top_groups(self):
        df = _synthetic_batch_df(300, seed=6)
        a = ErgoAnalyzer(df)
        top = a.top_groups("rula", "process_label", min_n=5, k=3)
        assert len(top) <= 3
        # sorted descending
        col = "mean"
        assert top[col].is_monotonic_decreasing


class TestExport:
    def test_writes_csvs(self, tmp_path):
        df = _synthetic_batch_df(120, seed=7)
        a = ErgoAnalyzer(df)
        out = export_report(a, tmp_path, group_specs=[["domain"], ["factory_id"]])
        # at least the agreement and group-difference files
        assert (out / "cross_method_agreement.csv").exists()
        assert (out / "group_difference_domain.csv").exists()
        # per-group files
        files = list(out.glob("*.csv"))
        names = [f.name for f in files]
        assert any("by_domain" in n for n in names)
        assert any("by_factory_id" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
