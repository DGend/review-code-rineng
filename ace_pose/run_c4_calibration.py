"""Fit and evaluate per-domain temperature calibration on MC distributions.

Reads `summary_with_mc.csv` + `mc_distributions.npz` produced by
`run_c3_mc`, builds `RiskDistribution` records per (image, method),
stratified-splits into train/val, fits per-(domain, method) temperature,
and reports Brier + ECE before vs after.

Important caveat: we have no human expert labels in this dataset.
The "label" used here is the *deterministic* ergo_v2 score on the
un-perturbed pose (the headline `*_grand` / `owas_action` columns
in the summary). This validates whether MC distributions are biased
relative to the deterministic baseline — it is NOT a clinical
calibration against expert raters. The latter requires a separate
labelling effort (flagged for the manuscript's limitations section).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.confidence_mc import RiskDistribution
from ace_pose.domain_calibration import (
    TemperatureCalibrator, evaluate_calibration, stratified_split,
)

log = logging.getLogger("ace_pose.run_c4_calibration")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


METHOD_RANGE = {"owas": (1, 4), "rula": (1, 7), "reba": (1, 15)}
PROB_KEY = {"owas": "owas_probs", "rula": "rula_probs", "reba": "reba_probs"}
LABEL_COL = {"owas": "owas_action", "rula": "rula_grand", "reba": "reba_grand"}


def _build_records(df: pd.DataFrame, npz_data) -> list[tuple]:
    """Build calibration records `(domain, RiskDistribution, label)`."""
    file_stems = npz_data["file_stem"]
    stem_to_row = {s: i for i, s in enumerate(file_stems)}

    records = []
    for _, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        i = stem_to_row.get(row["file_stem"])
        if i is None:
            continue
        domain = row["domain"]
        for method, (lo, hi) in METHOD_RANGE.items():
            probs = npz_data[PROB_KEY[method]][i]
            if probs.sum() < 1e-6:
                continue
            label = row[LABEL_COL[method]]
            if pd.isna(label):
                continue
            classes = np.arange(lo, hi + 1, dtype=int)
            dist = RiskDistribution(
                method=method,
                classes=classes,
                probs=probs / probs.sum(),
                n_samples=int(row.get("mc_n_effective", 0)),
            )
            records.append((domain, dist, int(label)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    df = pd.read_csv(batch_dir / "summary_with_mc.csv")
    npz = np.load(batch_dir / "mc_distributions.npz", allow_pickle=True)

    log.info("Building calibration records...")
    records = _build_records(df, npz)
    log.info("Built %d (domain, distribution, label) records", len(records))

    # Stratified train/val split keyed by (domain, method)
    train, val = stratified_split(records, val_frac=args.val_frac, rng_seed=args.seed)
    log.info("train=%d  val=%d", len(train), len(val))

    # Baseline (no calibration)
    baseline_cal = TemperatureCalibrator()  # T=1 everywhere
    baseline_reports = evaluate_calibration(val, baseline_cal)
    log.info("--- BASELINE (T=1) ---")
    for (domain, method), rep in sorted(baseline_reports.items()):
        log.info("  %s/%s  n=%d  Brier=%.4f  ECE=%.4f",
                 domain, method, rep.n, rep.brier_before, rep.ece_before)

    # Fit per (domain, method)
    log.info("Fitting per-(domain, method) temperature on train set...")
    fitted = TemperatureCalibrator().fit(train)
    log.info("Fitted temperatures: %s",
             {f"{d}/{m}": round(T, 3) for (d, m), T in fitted.by_key.items()})

    # Evaluate on val
    val_reports = evaluate_calibration(val, fitted)
    log.info("--- AFTER CALIBRATION (val) ---")
    rows = []
    for (domain, method), rep in sorted(val_reports.items()):
        T = fitted.get_T(domain, method)
        log.info("  %s/%s  n=%d  T=%.3f  Brier %.4f→%.4f  ECE %.4f→%.4f",
                 domain, method, rep.n, T,
                 rep.brier_before, rep.brier_after,
                 rep.ece_before, rep.ece_after)
        rows.append({
            "domain": domain, "method": method, "n": rep.n,
            "temperature": T,
            "brier_before": rep.brier_before, "brier_after": rep.brier_after,
            "brier_delta": rep.brier_before - rep.brier_after,
            "ece_before": rep.ece_before, "ece_after": rep.ece_after,
            "ece_delta": rep.ece_before - rep.ece_after,
        })

    report_df = pd.DataFrame(rows)
    out_csv = batch_dir / "c4_calibration_report.csv"
    report_df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # Aggregate stats
    log.info("--- AGGREGATE ---")
    by_method = report_df.groupby("method").agg({
        "brier_before": "mean", "brier_after": "mean",
        "ece_before": "mean", "ece_after": "mean",
    }).round(4)
    log.info("\n%s", by_method.to_string())

    # Save the fitted calibrator
    json.dump(
        {f"{k[0]}/{k[1]}": v for k, v in fitted.by_key.items()},
        open(batch_dir / "c4_temperatures.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
