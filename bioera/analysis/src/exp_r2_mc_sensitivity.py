#!/usr/bin/env python
# PWW-DISCLOSURE: public
"""exp-r2-mc-sensitivity: sensitivity of the Monte Carlo score distribution and its
temperature calibration to the perturbation scale (sigma_base) and sample count (S).

Design is pre-registered in paper/targets/targets.md. One parameter varies at a time:
sigma_base in {0.01,0.02,0.04,0.08,0.16} at S=128, and S in {32,64,128,256,512} at sigma_base=0.04.
Per setting and (domain, method) cell: mode-match rate, 5-95% interval coverage, ECE before,
fitted T and ECE after on the pre-specified grid and on the extended grid (0.1 up).
Scoring/MC/ECE/T-fit are the existing ace_pose functions; this file only loops over settings.
Nothing is written under ace_pose/. Outputs go to $PWW_RESULT_DIR.

v2 (targets.md "검수 반영 개정"): T is fitted and ECE-after measured on WORKER-held-out folds, the same
folds as exp_r2_indep_calibration (imported: build_cohort, make_folds, fold_eval; train and validation workers
asserted disjoint). Target stays the pipeline's own deterministic score (self-consistency sensitivity).
The first run's image-level random split is kept as reference columns with suffix _leaky.

Exploratory additions after the internal review (explor_settings.csv, explor_seeds.csv, metrics 'explor/...'): seconds per image,
share of near ties (top-2 gap < 0.1), mode match on the annotated images, seeds EXPLOR_SEEDS (default 0,1,2) at the current setting.
--limit N takes a random subsample of N poses (smoke test).

usage: python exp_r2_mc_sensitivity.py [--limit N] [--workers K]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.confidence_mc import McConfig, mc_score_all
from ace_pose.domain_calibration import TemperatureCalibrator, evaluate_calibration, stratified_split
from ace_pose.ergo_v2.context_policy import cohort_context
from ace_pose.major_joints import Major25Pose
from ace_pose.run_c4_calibration import _build_records
from pww_data import resolve

import exp_r2_indep_calibration as X  # cohort, folds, fold_eval (not re-implemented)

BATCH = Path(resolve("cohort_pose_cache"))
DEFAULT = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
EXTENDED = np.concatenate([[0.1, 0.2, 0.3, 0.4], DEFAULT, [7.5, 10, 15, 20, 30, 50, 100]])
THRESH = {"owas": 0.05, "rula": 0.07, "reba": 0.10}
LABEL = {"owas": "owas_action", "rula": "rula_grand", "reba": "reba_grand"}
# (sigma_base, S) settings; the shared current setting appears once
SETTINGS = [(s, 128) for s in (0.01, 0.02, 0.04, 0.08, 0.16)] + [(0.04, n) for n in (32, 64, 256, 512)]
CURRENT = (0.04, 128)
_DF = None


def sname(sig, S):
    return f"sigma{sig:g}_S{S}"


def mode_and_coverage(probs, y):
    """Per-pose: modal class == deterministic score y; 5-95 percentile interval contains y.
    Same definitions as src/mc_coverage.py. Classes are 1..len(probs)."""
    p = probs / probs.sum()
    cdf = np.cumsum(p)
    lo = np.searchsorted(cdf, 0.05) + 1
    hi = min(np.searchsorted(cdf, 0.95), len(p) - 1) + 1
    return int(np.argmax(p) + 1 == y), int(lo <= y <= hi)


def _init(df):
    global _DF
    _DF = df


def _mc_one(args):
    """MC distributions of one pose under one (sigma, S, seed) via the existing mc_score_all. Also returns the seconds spent in
    mc_score_all (wall clock of one image inside a busy pool, so only comparable across settings of the same run)."""
    i, sig, S, seed = args
    row = _DF.iloc[i]
    out = [np.zeros(4, np.float32), np.zeros(7, np.float32), np.zeros(15, np.float32)]
    npz = BATCH / "details" / f"{row.file_stem}.npz"
    if not npz.exists():
        return i, out, float("nan")
    d = np.load(npz)
    pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                       confidence=d["conf_m25"].astype(np.float64),
                       source=np.zeros(25, dtype=np.int8))
    t0 = time.perf_counter()
    try:
        res = mc_score_all(pose, config=McConfig(n_samples=S, sigma_fraction=sig, rng_seed=seed),
                           world_up=np.array([0.0, -1.0, 0.0]), walking=False,
                           **cohort_context(getattr(row, "process_label", None)),
                           domain=row.domain, process_id=int(getattr(row, "process_id", 0)) or None)
    except Exception:
        return i, out, float("nan")
    dt = time.perf_counter() - t0
    for a, dist in zip(out, (res.owas_dist, res.rula_dist, res.reba_dist)):
        for c in dist.classes:
            a[c - 1] = dist.probs[c - dist.classes[0]]
    return i, out, dt


def run_mc(pool, n, sig, S, seed=0):
    """Returns the probability arrays; probs['seconds'] = per-image seconds in mc_score_all (NaN when the pose failed)."""
    P = [np.zeros((n, k), np.float32) for k in (4, 7, 15)]
    sec = np.full(n, np.nan)
    for i, out, dt in pool.imap_unordered(_mc_one, [(i, sig, S, seed) for i in range(n)], chunksize=16):
        for a, o in zip(P, out):
            a[i] = o
        sec[i] = dt
    return dict(owas_probs=P[0], rula_probs=P[1], reba_probs=P[2], seconds=sec)


def top2_gap(p):
    """Sample-share difference between the most and second most frequent class (rows of p sum to 1)."""
    q = np.sort(p, axis=1)
    return q[:, -1] - q[:, -2]


def evaluate(df, probs, sig, S, worker, coh_stems):
    """Rows of the summary table for one setting. `worker` = worker_cells() result (v2 primary);
    the image-level split below is kept only as the _leaky reference columns."""
    npz = dict(file_stem=df["file_stem"].to_numpy(), **{k: v for k, v in probs.items() if k.endswith("_probs")})
    train, val = stratified_split(_build_records(df, npz), val_frac=0.2, rng_seed=0)
    rep = {}
    for g, grid in (("default", DEFAULT), ("extended", EXTENDED)):
        cal = TemperatureCalibrator().fit(train, grid=grid)
        rep[g] = (cal, evaluate_calibration(val, cal))
    rows = []
    for (dom, m), r in sorted(rep["default"][1].items()):
        sel = (df["domain"] == dom) & (df["status"] == "ok") & (probs[f"{m}_probs"].sum(1) > 1e-6)
        mc = np.array([mode_and_coverage(probs[f"{m}_probs"][i], int(df[LABEL[m]].iloc[i]))
                       for i in np.flatnonzero(sel.to_numpy()) if not pd.isna(df[LABEL[m]].iloc[i])])
        re = rep["extended"][1][(dom, m)]
        # exploratory (A6): near-tie share (top-2 gap < 0.1), mode match recounted on the images that have annotated keypoints
        P_m = probs[f"{m}_probs"]
        gap = top2_gap(P_m[sel.to_numpy()])
        ann_ix = np.flatnonzero((sel & df["file_stem"].isin(coh_stems) & df[LABEL[m]].notna()).to_numpy())
        rows.append(dict(setting=sname(sig, S), sigma_base=sig, S=S, domain=dom, method=m.upper(),
                         n_all=len(mc), mode_match=mc[:, 0].mean(), coverage=mc[:, 1].mean(),
                         **worker.get((dom, m.upper()), {}),
                         n_val_leaky=r.n, ece_before_leaky=r.ece_before,
                         T_default_leaky=rep["default"][0].get_T(dom, m), ece_after_leaky=r.ece_after,
                         T_extended_leaky=rep["extended"][0].get_T(dom, m), ece_after_ext_leaky=re.ece_after,
                         thresh=THRESH[m],
                         explor_near_tie_share=float((gap < 0.1).mean()), explor_n_annotated=len(ann_ix),
                         explor_mode_match_annotated=float(np.mean([mode_and_coverage(P_m[i], int(df[LABEL[m]].iloc[i]))[0] for i in ann_ix])) if len(ann_ix) else float("nan"),
                         explor_sec_per_image_mean=float(np.nanmean(probs["seconds"])), explor_sec_per_image_median=float(np.nanmedian(probs["seconds"]))))
    return rows


def worker_key(coh):
    return np.array([f"{a}_{b}" for a, b in zip(coh.factory_id, coh.worker_id)])


def assert_worker_disjoint(folds, wk):
    """Every fold: validation non-empty and no worker appears in both train and validation."""
    for dom, fl in folds.items():
        for k, (tr, va) in enumerate(fl):
            assert len(va) and len(tr), f"{dom} fold {k}: empty train or validation"
            both = set(wk[tr]) & set(wk[va])
            assert not both, f"{dom} fold {k}: workers in both train and validation: {sorted(both)[:3]}"


def worker_cells(df, probs, coh, folds):
    """Worker-held-out cells {(domain, METHOD): metrics} on the indep cohort rows of df. Target = df's deterministic score."""
    pos = {s: i for i, s in enumerate(df["file_stem"])}
    ix = np.array([pos[s] for s in coh.file_stem])
    out = {}
    for dom in X.DOMAINS:
        di = np.flatnonzero(coh.domain.to_numpy() == dom)
        for m in X.METHODS:
            P, yv = probs[f"{m}_probs"][ix], df[LABEL[m]].to_numpy()[ix]
            ok = (P.sum(1) > 1e-6) & ~pd.isna(yv)
            y = np.where(ok, yv, 0).astype(int)
            ds = X.dists_of(np.where(ok[:, None], P, 1.0), m)
            fl = [([i for i in tr if ok[i]], [i for i in va if ok[i]]) for tr, va in folds[dom]]
            dv = [i for i in di if ok[i]]
            r = {"n_cohort": len(dv), "n_folds": len(fl),
                 "ece_before": X.expected_calibration_error(X.cal_dists([ds[i] for i in dv], 1.0), [int(y[i]) for i in dv], X.N_BINS)}
            for g, grid in (("default", DEFAULT), ("ext", EXTENDED)):
                res = [X.fold_eval(ds, y, dom, tr, va, grid) for tr, va in fl]
                Ts, ec = np.array([x[0] for x in res]), np.array([x[1] for x in res])
                r.update({f"T_{g}_{k}": float(getattr(Ts, k)()) for k in ("mean", "min", "max")})
                r.update({f"ece_after_{g}_{k}": float(getattr(ec, k)()) for k in ("mean", "min", "max")})
            out[(dom, m.upper())] = r
    return out


def seed_cells(seed, wc):
    """Self-consistency ECE (before calibration, and after worker-held-out calibration on the pre-specified grid) per cell for one seed."""
    return [dict(seed=seed, domain=d, method=m, ece_before=r["ece_before"], ece_after_default_mean=r["ece_after_default_mean"]) for (d, m), r in wc.items()]


def robustness(tab, ece="ece_after_default_mean"):
    """Pre-registered criterion: for sigma 0.02-0.08 (S=128) and S>=64 (sigma 0.04),
    (a) every cell `ece` <= threshold, (b) |mode_match - current| <= 0.02. Returns dict.
    v2: ece = worker-held-out mean on the pre-specified grid; the leaky run used ece_after."""
    cur = tab[(tab.sigma_base == CURRENT[0]) & (tab.S == CURRENT[1])].set_index(["domain", "method"])
    out = {"a_breaks": [], "b_breaks": []}
    for (sig, S), g in tab.groupby(["sigma_base", "S"]):
        if not ((0.02 <= sig <= 0.08 and S == 128) or (sig == 0.04 and S >= 64)):
            continue
        for r in g.itertuples():
            if getattr(r, ece) > r.thresh:
                out["a_breaks"].append(f"{sname(sig, S)}/{r.domain}/{r.method}: {ece} {getattr(r, ece):.4f} > {r.thresh}")
            d = r.mode_match - cur.loc[(r.domain, r.method), "mode_match"]
            if abs(d) > 0.02:
                out["b_breaks"].append(f"{sname(sig, S)}/{r.domain}/{r.method}: mode_match diff {100*d:+.2f} pp")
    out["a_pass"], out["b_pass"] = not out["a_breaks"], not out["b_breaks"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "bioera/analysis/result/exp-r2-mc-sensitivity-v2"))
    out.mkdir(parents=True, exist_ok=True)
    # labels and per-pose context come from the cohort summary (read only)
    df = pd.read_csv(BATCH / "summary_with_mc.csv")
    df = df[df["status"] == "ok"].reset_index(drop=True) if "status" in df.columns else df
    if a.limit:
        df = df.sample(a.limit, random_state=0).reset_index(drop=True)   # random subsample: head() covers too few workers for the folds
    # worker-held-out folds: identical to exp_r2_indep_calibration (cohort filter, seed, fold builder)
    coh = X.build_cohort()
    coh = coh[coh.file_stem.isin(set(df["file_stem"]))].reset_index(drop=True)  # only differs under --limit
    folds = X.make_folds(coh, X._load_exp012())
    wk = worker_key(coh)
    assert_worker_disjoint(folds, wk)
    print(f"PWW_PROGRESS 0/{len(SETTINGS)} folds ok: cohort {len(coh)}, worker-disjoint in all "
          f"{sum(len(v) for v in folds.values())} folds", flush=True)
    rows, seed_rows, zero_prob, t0 = [], [], {}, time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(df,)) as pool:
        for k, (sig, S) in enumerate(SETTINGS):
            probs = run_mc(pool, len(df), sig, S)
            zero_prob[sname(sig, S)] = int((probs["owas_probs"].sum(1) <= 1e-6).sum())   # images whose MC failed or had no cached pose
            wc = worker_cells(df, probs, coh, folds)
            rows += evaluate(df, probs, sig, S, wc, set(coh.file_stem))
            if (sig, S) == CURRENT:  # sanity: reproduces the cached production run?
                seed_rows += seed_cells(0, wc)
                for sd in [x for x in X.EXPLOR_SEEDS if x != 0]:     # exploratory seed repetition (A5); seed 0 is the main run above
                    seed_rows += seed_cells(sd, worker_cells(df, run_mc(pool, len(df), sig, S, sd), coh, folds))
                z = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
                idx = {s: i for i, s in enumerate(z["file_stem"])}
                j = [idx[s] for s in df["file_stem"]]
                cache_diff = max(float(np.abs(probs[f"{m}_probs"] - z[f"{m}_probs"][j]).max()) for m in LABEL)
            print(f"PWW_PROGRESS {k + 1}/{len(SETTINGS)} {sname(sig, S)} {time.time() - t0:.0f}s", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary_table.csv", index=False)
    met = {f"{r.setting}/{r.domain}/{r.method}/{c}": float(getattr(r, c))
           for r in tab.itertuples() for c in ("mode_match", "coverage", "ece_before", "ece_before_leaky", "n_val_leaky", "n_all", "n_cohort", "n_folds",
                                              *[c for c in tab.columns if c.startswith(("T_", "ece_after"))])}
    crit = robustness(tab)
    crit["ext_grid"] = robustness(tab, "ece_after_ext_mean")   # information only; not the pre-registered grid
    crit["leaky_reference"] = robustness(tab, "ece_after_leaky")
    met.update({"n_poses": len(df), "cache_vs_current_max_abs_prob_diff": cache_diff,
                "robust_a_pass": float(crit["a_pass"]), "robust_b_pass": float(crit["b_pass"])})
    # exploratory additions (A5 seeds, A6 timing / near ties / annotated subset): CSV + 'explor/' metrics
    xcols = [c for c in tab.columns if c.startswith("explor_")]
    tab[["setting", "sigma_base", "S", "domain", "method", *xcols]].to_csv(out / "explor_settings.csv", index=False)
    for r in tab.itertuples():
        for c in xcols:
            met[f"explor/{r.setting}/{r.domain}/{r.method}/{c[7:]}"] = float(getattr(r, c))
    met.update({f"explor/mc_zero_prob_images/{k}": v for k, v in zero_prob.items()})   # the MC except-clauses swallow failures: count them
    sd_tab = pd.DataFrame(seed_rows)
    sd_tab.to_csv(out / "explor_seeds.csv", index=False)
    for (d, m), g in sd_tab.groupby(["domain", "method"]):
        for c in ("ece_before", "ece_after_default_mean"):
            met[f"explor/seeds/{d}/{m}/{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")   # ddof=1 over the seeds
            met.update({f"explor/seeds/{d}/{m}/{c}_seed{sd}": float(v) for sd, v in zip(g.seed, g[c])})
    (out / "metrics.json").write_text(json.dumps(met, indent=1))
    (out / "robustness.json").write_text(json.dumps(crit, indent=1, ensure_ascii=False))
    print(json.dumps(crit, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
