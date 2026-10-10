"""Full-cohort (6,007-pose) per-vertebral-disc compression and
sex/age-coupled herniation risk, plus the demographic effective-load
adjustment — the empirical replacement for the 9-example-worker
illustrations of §3.5.4 / §3.5.5.

**Scale consistency (resolves the reviewer's length-scale concern).**
Every pose is scaled with its *own* `biomech_length_scale_m` taken
verbatim from the validated batch summary, i.e. the exact scale that
produced the paper's `biomech_L5S1_Nm` column — not the previous
`1.72 / stature` heuristic.  We additionally report the agreement
between the dense-spine L5/S1 moment and the independently-validated
`biomech_L5S1_Nm` as an internal-consistency check.

**Empirical, not combinatorial.**  The seven demographic profiles
remain *reference* profiles (per-worker sex/age/mass are not annotated
in the Busan dataset), but they are now applied to the **real
posture distribution of all 6,007 poses** (and broken down by the
three work domains), so the risk-band fractions are a cohort
statistic rather than a 9-worker toy.

Outputs (in `batch/analysis/`):
  * `cohort_disc_population.csv`     — per-domain compression stats
  * `cohort_disc_risk_by_profile.csv`— profile × risk-band %, 6,007 poses
  * `cohort_demographic_by_profile.csv` — profile × joint × risk %
  * `cohort_disc_population.png`     — L5/S1 compression distribution
                                       per domain, NIOSH AL/MPL lines
  * `cohort_disc_risk_by_profile.png`— stacked risk bars over 6,007
  * `cohort_demographic_risk.png`    — per-joint demographic stratific.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.dense_spine import estimate_dense_spine
from ace_pose.disc_load import (
    disc_compression_N, disc_tolerance_N, disc_risk_band,
    trunk_flexion_deg_3d, HERNIATION_LEVELS,
    NIOSH_AL_N, NIOSH_MPL_N,
)
from ace_pose.demographic_load import (
    DEFAULT_PROFILES, effective_load_rate, risk_category, RISK_COLOR,
)

log = logging.getLogger("ace_pose.run_cohort_biomech")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

RISK_LEVELS = ["safe", "caution", "elevated", "surgical"]
DEMO_RISK = ["safe", "caution", "elevated", "surgical"]
BODY_MASS_KG = 70.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=str)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap number of poses (0 = all)")
    args = ap.parse_args()
    batch_dir = Path(args.batch)
    out_dir = batch_dir / "analysis"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for cand in ("summary_with_xw.csv", "summary_with_mc.csv",
                 "summary_with_pred.csv", "summary.csv"):
        p = batch_dir / cand
        if p.exists():
            sum_path = p
            break
    df = pd.read_csv(sum_path)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("Cohort: %d poses from %s", len(df), sum_path.name)

    det = batch_dir / "details"
    rec = []
    n_bad = 0
    for i, r in df.iterrows():
        npz = det / f"{r['file_stem']}.npz"
        if not npz.exists():
            n_bad += 1
            continue
        try:
            d = np.load(npz)
            kp = d["kpts_m25"]; cf = d["conf_m25"]
        except Exception:
            n_bad += 1
            continue
        ls = float(r["biomech_length_scale_m"])
        if not np.isfinite(ls) or ls <= 0:
            n_bad += 1
            continue
        spine = estimate_dense_spine(kp, cf, body_mass_kg=BODY_MASS_KG,
                                     length_scale_m=ls)
        flex = trunk_flexion_deg_3d(kp, cf)
        comp = disc_compression_N(spine, BODY_MASS_KG, flex)
        lv_idx = {lv: j for j, lv in enumerate(spine.levels)}
        dense_l5 = float(spine.moments_Nm[lv_idx["L5/S1"]])
        rec.append({
            "domain": r["domain"],
            "flex": flex,
            "C_L5S1": comp.get("L5/S1", 0.0),
            "C_L3": comp.get("L3", 0.0),
            "C_L1": comp.get("L1", 0.0),
            "C_T12": comp.get("T12", 0.0),
            "dense_L5S1_Nm": dense_l5,
            "sum_L5S1_Nm": float(r["biomech_L5S1_Nm"]),
            "sum_sho_Nm": float(r.get("biomech_shoulder_max_Nm", np.nan)),
            "sum_knee_Nm": float(r.get("biomech_knee_max_Nm", np.nan)),
        })
        if (i + 1) % 1000 == 0:
            log.info("  ...%d / %d", i + 1, len(df))

    cf_df = pd.DataFrame(rec)
    log.info("Processed %d poses (%d skipped)", len(cf_df), n_bad)
    cf_df.to_csv(out_dir / "cohort_disc_perpose.csv", index=False)

    # ---- internal consistency vs validated biomech_L5S1_Nm
    m = cf_df["sum_L5S1_Nm"] > 1e-6
    rr = (cf_df.loc[m, "dense_L5S1_Nm"] / cf_df.loc[m, "sum_L5S1_Nm"])
    corr = float(np.corrcoef(cf_df.loc[m, "dense_L5S1_Nm"],
                             cf_df.loc[m, "sum_L5S1_Nm"])[0, 1])
    log.info("Internal consistency dense vs validated L5/S1: "
             "r=%.4f  median ratio=%.3f", corr, float(rr.median()))

    # ---- profile-independent population stats (per domain + overall)
    pop_rows = []
    for dom, g in [("ALL", cf_df)] + list(cf_df.groupby("domain")):
        for lvl, col in [("L5/S1", "C_L5S1"), ("L3", "C_L3"),
                         ("L1", "C_L1"), ("T12", "C_T12")]:
            v = g[col].to_numpy()
            pop_rows.append({
                "domain": dom, "level": lvl, "n": len(v),
                "median_N": np.median(v), "p90_N": np.percentile(v, 90),
                "p95_N": np.percentile(v, 95),
                "p99_N": np.percentile(v, 99), "max_N": v.max(),
                "pct_over_AL": 100.0 * np.mean(v >= NIOSH_AL_N),
                "pct_over_MPL": 100.0 * np.mean(v >= NIOSH_MPL_N),
            })
    pop = pd.DataFrame(pop_rows)
    pop.to_csv(out_dir / "cohort_disc_population.csv", index=False)
    log.info("\nPopulation L5/S1 (ALL): \n%s",
             pop[(pop.domain == "ALL")].to_string(index=False))

    # ---- per-profile risk-band % over all 6,007 poses (4 lumbar discs)
    prof_rows = []
    comp_cols = {"L5/S1": "C_L5S1", "L3": "C_L3",
                 "L1": "C_L1", "T12": "C_T12"}
    for prof in DEFAULT_PROFILES:
        for scope, g in [("ALL", cf_df)] + list(cf_df.groupby("domain")):
            counts = {k: 0 for k in RISK_LEVELS}
            tot = 0
            for lvl, col in comp_cols.items():
                tol = disc_tolerance_N(lvl, prof.sex, prof.age)
                ratios = g[col].to_numpy() / max(tol, 1e-6)
                for x in ratios:
                    counts[disc_risk_band(float(x))] += 1
                    tot += 1
            row = {"profile": prof.label, "sex": prof.sex,
                   "age": prof.age, "scope": scope, "n_cells": tot}
            for k in RISK_LEVELS:
                row[k] = 100.0 * counts[k] / max(tot, 1)
            prof_rows.append(row)
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(out_dir / "cohort_disc_risk_by_profile.csv",
                   index=False)
    log.info("\nPer-profile disc risk %% (ALL, 6,007×4 discs):\n%s",
             prof_df[prof_df.scope == "ALL"][
                 ["profile"] + RISK_LEVELS].round(1).to_string(index=False))

    # ---- demographic effective-load (validated summary joint moments)
    JMAP = {"L5S1": "sum_L5S1_Nm", "shoulder": "sum_sho_Nm",
            "knee": "sum_knee_Nm"}
    demo_rows = []
    for prof in DEFAULT_PROFILES:
        for jname, col in JMAP.items():
            v = cf_df[col].to_numpy()
            v = v[np.isfinite(v)]
            cats = {k: 0 for k in DEMO_RISK}
            for x in v:
                lr = effective_load_rate(float(x), jname,
                                         prof.sex, prof.age)
                cats[risk_category(lr)] += 1
            row = {"profile": prof.label, "joint": jname, "n": len(v)}
            for k in DEMO_RISK:
                row[k] = 100.0 * cats[k] / max(len(v), 1)
            demo_rows.append(row)
    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(out_dir / "cohort_demographic_by_profile.csv",
                   index=False)

    # ===================== FIGURES =====================
    # 1) L5/S1 compression distribution per domain + NIOSH lines
    fig, ax = plt.subplots(figsize=(10, 5.5))
    doms = ["manufacturing", "eldercare", "logistics"]
    colors = {"manufacturing": "#2563eb", "eldercare": "#16a34a",
              "logistics": "#d97706"}
    bins = np.linspace(0, 8000, 60)
    for dm in doms:
        v = cf_df.loc[cf_df.domain == dm, "C_L5S1"].clip(0, 8000)
        if len(v) == 0:
            continue
        ax.hist(v, bins=bins, histtype="step", linewidth=2.0,
                color=colors[dm], label=f"{dm} (n={len(v)})", density=True)
    ax.axvline(NIOSH_AL_N, color="#f59e0b", linestyle="--", linewidth=2,
               label="NIOSH AL 3400 N")
    ax.axvline(NIOSH_MPL_N, color="#dc2626", linestyle="--", linewidth=2,
               label="NIOSH MPL 6400 N")
    ax.set_xlabel("L5/S1 disc compression (N)")
    ax.set_ylabel("density")
    ax.set_title("Cohort L5/S1 disc-compression distribution by work "
                 f"domain (n = {len(cf_df)} poses)\n"
                 "scale = validated biomech_length_scale_m per pose",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cohort_disc_population.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(fig_dir / "cohort_disc_population.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # 2) per-profile stacked risk bars over the full cohort
    pl = [p.label for p in DEFAULT_PROFILES]
    sub = (prof_df[prof_df.scope == "ALL"].set_index("profile")
           .reindex(pl))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(pl))
    for k in RISK_LEVELS:
        ax.bar(pl, sub[k].values, bottom=bottom, color=RISK_COLOR[k],
               label=k, edgecolor="white")
        bottom += sub[k].values
    ax.set_ylabel("% of (6,007 poses × 4 lumbar discs) cells")
    ax.set_ylim(0, 100)
    ax.set_title("Empirical disc-herniation risk distribution per "
                 "demographic profile\n"
                 f"reference profiles applied to all {len(cf_df)} "
                 "cohort postures (4 clinical lumbar discs)",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, -0.08),
              frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "cohort_disc_risk_by_profile.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(fig_dir / "cohort_disc_risk_by_profile.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # 3) demographic per-joint stratification (L5S1 column)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    s2 = (demo_df[demo_df.joint == "L5S1"].set_index("profile")
          .reindex(pl))
    bottom = np.zeros(len(pl))
    for k in DEMO_RISK:
        ax.bar(pl, s2[k].values, bottom=bottom, color=RISK_COLOR[k],
               label=k, edgecolor="white")
        bottom += s2[k].values
    ax.set_ylabel("% of 6,007 poses")
    ax.set_ylim(0, 100)
    ax.set_title("Empirical L5/S1 effective-load risk per demographic "
                 f"profile (validated moments, n = {len(cf_df)})",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, -0.08),
              frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "cohort_demographic_risk.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(fig_dir / "cohort_demographic_risk.pdf",
                bbox_inches="tight")
    plt.close(fig)

    log.info("internal-consistency r=%.4f  median(dense/validated)=%.3f",
             corr, float(rr.median()))
    log.info("Done. CSVs + 3 figures in %s", out_dir)


if __name__ == "__main__":
    main()
