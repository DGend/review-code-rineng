"""Run the modular analysis on the current batch and produce per-location
reports.

Usage:
    python -m ace_pose.run_analysis --batch ace_pose/_runs/batch_v1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace_pose.analysis import (
    ErgoAnalyzer, FACTORY_LABEL, export_report,
)

log = logging.getLogger("ace_pose.run_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def add_location_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `location` column (factory friendly name) when factory_id present."""
    df = df.copy()
    if "factory_id" in df.columns:
        df["location"] = df["factory_id"].map(FACTORY_LABEL).fillna("Unknown")
    return df


def make_pivot(s, method: str) -> pd.DataFrame | None:
    """Wide-format pivot of action distributions for the manuscript."""
    if method not in s.action_dist:
        return None
    return s.action_dist[method].reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str,
                        help="Path to batch run dir containing summary.csv")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: <batch>/analysis)")
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    df = pd.read_csv(batch_dir / "summary.csv")
    df = add_location_label(df)
    log.info("Loaded %d rows from %s", len(df), batch_dir)
    log.info("Domains: %s", dict(df["domain"].value_counts()))
    log.info("Locations: %s", dict(df["location"].value_counts()))

    out_dir = Path(args.out) if args.out else batch_dir / "analysis"
    analyzer = ErgoAnalyzer(df)

    # Three location groupings the manuscript needs:
    #   1. domain (3-way)
    #   2. location (factory friendly name, 7-way)
    #   3. location × process (per-location, per-process — paper table)
    group_specs = [
        ["domain"],
        ["location"],
        ["location", "process_label"],
        ["process_label"],
    ]
    export_report(analyzer, out_dir, group_specs=group_specs)

    # Compact "headline" tables to stdout
    log.info("=" * 60)
    log.info("Headline: mean RULA by location")
    by_loc = analyzer.by(["location"])
    log.info("\n%s", by_loc.score_stats["rula"].round(3).to_string())

    log.info("Headline: mean REBA by location")
    log.info("\n%s", by_loc.score_stats["reba"].round(3).to_string())

    log.info("Headline: OWAS action-category distribution by location")
    log.info("\n%s", by_loc.action_dist["owas"].to_string())

    # Top 10 risky processes (by RULA mean, min 30 samples)
    log.info("=" * 60)
    log.info("Top-10 RULA processes (min n=30):")
    top_rula = analyzer.top_groups("rula", "process_label", min_n=30, k=10)
    log.info("\n%s", top_rula.round(3).to_string())

    log.info("Top-10 REBA processes (min n=30):")
    top_reba = analyzer.top_groups("reba", "process_label", min_n=30, k=10)
    log.info("\n%s", top_reba.round(3).to_string())

    # Cross-method agreement
    log.info("=" * 60)
    agr = analyzer.cross_method_agreement()
    log.info("Cross-method agreement:\n%s", agr.pairs.round(3).to_string(index=False))

    # Statistical tests by domain
    log.info("Group-difference tests (domain):")
    for method in ("owas", "rula", "reba"):
        gd = analyzer.group_difference(method, "domain")
        log.info("  %s  KW H=%s p=%s   chi2=%s p=%s",
                 method,
                 _fmt(gd.kw.get("H")), _fmt(gd.kw.get("p")),
                 _fmt(gd.chi2.get("chi2")), _fmt(gd.chi2.get("p")))

    # Group-difference tests by location (more granular)
    log.info("Group-difference tests (location):")
    for method in ("owas", "rula", "reba"):
        gd = analyzer.group_difference(method, "location")
        log.info("  %s  KW H=%s p=%s   chi2=%s p=%s",
                 method,
                 _fmt(gd.kw.get("H")), _fmt(gd.kw.get("p")),
                 _fmt(gd.chi2.get("chi2")), _fmt(gd.chi2.get("p")))

    # Confidence-stratified summary
    log.info("=" * 60)
    log.info("Confidence-stratified RULA mean (by domain):")
    for band, summ in analyzer.confidence_strata(low=0.5, high=0.7).items():
        if "rula" not in summ.score_stats:
            continue
        means = summ.score_stats["rula"]["mean"]
        log.info("  [%s] %s", band, means.to_dict())

    # Dump a tidy summary CSV for the manuscript appendix
    headline = {
        "n_total": len(df),
        "n_manufacturing": int((df["domain"] == "manufacturing").sum()),
        "n_eldercare": int((df["domain"] == "eldercare").sum()),
        "n_logistics": int((df["domain"] == "logistics").sum()),
        "rula_mean": float(df["rula_grand"].mean()),
        "reba_mean": float(df["reba_grand"].mean()),
        "owas_mean_action": float(df["owas_action"].mean()),
        "kpt_conf_mean": float(df["kpt_conf_mean"].mean()),
    }
    with open(out_dir / "headline_summary.json", "w") as f:
        json.dump(headline, f, indent=2)
    log.info("Wrote %s", out_dir / "headline_summary.json")


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if abs(x) < 1e-4:
            return f"{x:.2e}"
        return f"{x:.3f}"
    return str(x)


if __name__ == "__main__":
    main()
