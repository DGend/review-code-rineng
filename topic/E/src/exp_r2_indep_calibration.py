# PWW-DISCLOSURE: public
"""exp-r2-indep-calibration: calibration against a target that is independent of the predicted pose.

Reviewer objection: the Monte Carlo (MC) distribution is centred on the predicted pose and the
calibration target is the same scorer applied to that pose, so ECE can improve through
self-consistency alone. Here the target is the grand score of the human-ANNOTATED keypoints, scored
with the same scorer and the same posture-only task context (cohort_context) the MC used.
Design and decision rule: topic/E/paper/targets/targets.md, section exp-r2-indep-calibration.

Primary MC = the cached manuscript distribution (ace_pose/_runs/batch_v1/mc_distributions.npz,
read-only). Sensitivity MC = the same sampler re-run on the pose restricted to the joints the
annotation has (ears, small toes, heels removed), so MC centre and target see the same joint set.
Temperature folds reuse topic/A/src/public/exp012 (manufacturing: 20 random worker-group 80/20 splits;
eldercare, logistics: leave-one-worker-out). Outputs go to $PWW_RESULT_DIR.
Exploratory (explor_*.csv, metrics 'explor/...'): worker bootstrap of the pooled ECE after worker-held-out calibration, per-image
calibrated confidences (explor_cal_rows.csv). SMOKE=<n> subsamples n images.
"""
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_action_agreement as EA  # noqa: E402  (label_pose, system_pose, WU, BATCH, JSON_DIR)
from ace_pose.confidence_mc import (  # noqa: E402
    McConfig, RiskDistribution, apply_temperature, brier_score, expected_calibration_error, mc_score_all)
from ace_pose.domain_calibration import TemperatureCalibrator  # noqa: E402
from ace_pose.ergo_v2.context_policy import cohort_context  # noqa: E402
from ace_pose.ergo_v2.pipeline import score_all  # noqa: E402
from pww_data import resolve  # noqa: E402

OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "topic/E/result/exp-r2-indep-calibration"))
BATCH = EA.BATCH = Path(resolve("cohort_pose_cache"))  # EA.system_pose reads EA.BATCH
ANNOT = Path(resolve("aihub_busan_annotations"))
METHODS = {"owas": (1, 4), "rula": (1, 7), "reba": (1, 15)}
DET_COL = {"owas": "owas_action", "rula": "rula_grand", "reba": "reba_grand"}
DOMAINS = ["manufacturing", "eldercare", "logistics"]
GRID_DEF = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])       # pre-specified (confidence_mc)
GRID_EXT = np.concatenate([[0.1, 0.2, 0.3, 0.4], GRID_DEF])             # extended 0.1-5
THRESH = {"owas": 0.05, "rula": 0.07, "reba": 0.10}                     # pre-specified ECE thresholds (supp_temperature_grid.py)
# POST-HOC exploratory grid (not the pre-registered criterion): extends beyond the T = 5 bound every fold hit
GRID_EXPLOR = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50], float)
EXPLOR_SEEDS = [int(x) for x in os.environ.get("EXPLOR_SEEDS", "0,1,2").split(",") if x]   # exploratory seed repetition (A5); seed 0 = main
T_SELF = 0.5                                                            # manuscript self-consistency temperature
N_BINS = 10
MC_CFG = McConfig(n_samples=128, sigma_fraction=0.04, rng_seed=0)       # same as run_c3_mc defaults


def interval_5_95(p: np.ndarray, lo_cls: int = 1) -> tuple[int, int]:
    """5th and 95th percentile classes of a class-probability vector (same rule as mc_coverage.py)."""
    cdf = np.cumsum(p)
    return lo_cls + int(np.searchsorted(cdf, 0.05)), lo_cls + int(min(np.searchsorted(cdf, 0.95), len(p) - 1))


def _load_exp012():
    """Import the worker-fold helpers from the published exp012 script (not re-implemented)."""
    spec = importlib.util.spec_from_file_location("exp012", ROOT / "topic/A/src/public/exp012_c4_worker_holdout_refit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _target_worker(args):
    """Annotated-keypoint grand scores, scored exactly as run_c3_mc scores a pose (cohort_context, bilateral)."""
    stem, label, row = args
    r = score_all(EA.label_pose(row), world_up=EA.WU, walking=False, **cohort_context(label))
    return stem, (r.owas.action_category, r.rula.grand_score, r.reba.grand_score)


def _mc_matched_worker(args):
    """Sensitivity: MC on the joint-restricted predicted pose, same call as run_c3_mc."""
    stem, label, domain, pid = args
    res = mc_score_all(EA.system_pose(stem, True), config=MC_CFG, world_up=EA.WU, walking=False,
                       **cohort_context(label), domain=domain, process_id=int(pid) or None)
    return stem, (res.owas_dist.probs, res.rula_dist.probs, res.reba_dist.probs)


def build_cohort():
    """Images with status ok, matched label person, and one annotation (same filter as eval_action_agreement)."""
    summ = pd.read_csv(BATCH / "summary_with_mc.csv")
    lab = pd.read_csv(ANNOT)
    lab["file_stem"] = lab.file_name.str[:-4]
    ok = summ[(summ.status == "ok") & summ.label_match_ok.astype(bool)]
    m = ok.merge(lab, left_on=["file_stem", "worker_id"], right_on=["file_stem", "worker"], how="inner", suffixes=("", "_lab"))
    return m.drop_duplicates("file_stem", keep=False).reset_index(drop=True)


def reliability_bins(conf, correct):
    """Per-bin (mean confidence, accuracy, count) with the same bin edges as expected_calibration_error."""
    edges = np.linspace(0, 1, N_BINS + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi + (1e-9 if hi >= 1 else 0))
        if mk.any():
            out.append((float(conf[mk].mean()), float(correct[mk].mean()), int(mk.sum())))
    return out


def dists_of(probs, method):
    cls = np.arange(METHODS[method][0], METHODS[method][1] + 1)
    return [RiskDistribution(method=method, classes=cls, probs=p / p.sum(), n_samples=0) for p in probs]


def cal_dists(dists, T):
    return [RiskDistribution(d.method, d.classes, apply_temperature(d.probs, T), d.n_samples) for d in dists]


def fold_eval(dists, y, domain, train, val, grid):
    """Fit T on train (TemperatureCalibrator), evaluate on val. Returns T, ECE, Brier, and val confidences/correctness."""
    cal = TemperatureCalibrator().fit([(domain, dists[i], int(y[i])) for i in train], grid=grid)
    T = cal.get_T(domain, dists[0].method)
    vd = cal_dists([dists[i] for i in val], T)
    vy = [int(y[i]) for i in val]
    conf = np.array([d.probs.max() for d in vd])
    corr = np.array([float(d.classes[np.argmax(d.probs)] == t) for d, t in zip(vd, vy)])
    return T, expected_calibration_error(vd, vy, N_BINS), brier_score(vd, vy), conf, corr


def make_folds(coh, e12):
    """Image-level worker-held-out folds per domain, using the exp012 definitions and seed."""
    wk = np.array([f"{a}_{b}" for a, b in zip(coh.factory_id, coh.worker_id)])
    rng = np.random.default_rng(e12.SEED)
    folds = {}
    for dom in DOMAINS:
        idx = [i for i, d in enumerate(coh.domain) if d == dom]
        if dom == "manufacturing":
            folds[dom] = [e12.group_split_manufacturing(idx, wk, rng) for _ in range(e12.N_MANUFACTURING_REPEATS)]
        else:
            folds[dom] = [([i for i in idx if wk[i] != w], [i for i in idx if wk[i] == w]) for w in sorted(set(wk[idx]))]
    return folds


def evaluate(coh, probs, target, folds, tag, metrics, keep_curves=None, keep_cal=None):
    """All cell metrics for one (MC source, target) pair. probs[m]: (N, K) arrays; target[m]: (N,) ints."""
    rows = {}
    for dom in DOMAINS:
        di = np.where(coh.domain.to_numpy() == dom)[0]
        for m in METHODS:
            ds = dists_of(probs[m], m)
            y = target[m]
            sub = [ds[i] for i in di]
            sy = [int(y[i]) for i in di]
            r = {"n": len(di)}
            r["mode_acc"] = float(np.mean([int(d.mode) == t for d, t in zip(sub, sy)]))
            r["coverage"] = float(np.mean([lo <= t <= hi for (lo, hi), t in
                                           ((interval_5_95(d.probs, METHODS[m][0]), t) for d, t in zip(sub, sy))]))
            for name, T in (("a", 1.0), ("b", T_SELF)):
                cd = cal_dists(sub, T)
                r[f"ece_{name}"] = expected_calibration_error(cd, sy, N_BINS)
                r[f"brier_{name}"] = brier_score(cd, sy)
            for gname, grid in (("def", GRID_DEF), ("ext", GRID_EXT)):
                res = [fold_eval(ds, y, dom, tr, va, grid) for tr, va in folds[dom]]
                Ts, ec, br = (np.array([x[k] for x in res]) for k in (0, 1, 2))
                r.update({f"T_{gname}_mean": float(Ts.mean()), f"T_{gname}_min": float(Ts.min()), f"T_{gname}_max": float(Ts.max()),
                          f"ece_c_{gname}_mean": float(ec.mean()), f"ece_c_{gname}_min": float(ec.min()),
                          f"ece_c_{gname}_max": float(ec.max()), f"brier_c_{gname}_mean": float(br.mean()),
                          "n_folds": len(res)})
                if gname == "def" and keep_cal is not None:
                    keep_cal[(dom, m, "def")] = res
                if gname == "def" and keep_curves is not None:
                    keep_curves[(dom, m)] = (reliability_bins(np.array([d.probs.max() for d in cal_dists(sub, 1.0)]),
                                                              np.array([float(d.mode == t) for d, t in zip(sub, sy)])),
                                             reliability_bins(np.concatenate([x[3] for x in res]), np.concatenate([x[4] for x in res])))
            # POST-HOC exploratory: same worker folds, wider grid; also top-class confidence before/after scaling
            res = [fold_eval(ds, y, dom, tr, va, GRID_EXPLOR) for tr, va in folds[dom]]
            if keep_cal is not None:
                keep_cal[(dom, m, "explor")] = res
            Ts, ec, br = (np.array([x[k] for x in res]) for k in (0, 1, 2))
            r.update({"explor_T_mean": float(Ts.mean()), "explor_T_min": float(Ts.min()), "explor_T_max": float(Ts.max()),
                      "explor_ece_mean": float(ec.mean()), "explor_ece_min": float(ec.min()), "explor_ece_max": float(ec.max()),
                      "explor_brier_mean": float(br.mean()), "explor_T_at_upper_bound": bool((Ts >= GRID_EXPLOR.max()).any()),
                      "explor_frac_folds_at_upper_bound": float((Ts >= GRID_EXPLOR.max()).mean()),
                      "explor_acc_val": float(np.concatenate([x[4] for x in res]).mean()),
                      "explor_conf_after_mean": float(np.concatenate([x[3] for x in res]).mean()),
                      "explor_conf_raw_mean": float(np.mean([d.probs.max() for d in sub])),
                      "explor_pass": bool(ec.mean() <= THRESH[m])})
            r["thresh"] = THRESH[m]
            r["pass_def"] = bool(r["ece_c_def_mean"] <= THRESH[m])
            r["pass_ext"] = bool(r["ece_c_ext_mean"] <= THRESH[m])
            r["b_worse_than_a"] = bool(r["ece_b"] > r["ece_a"])
            rows[(dom, m)] = r
            for k, v in r.items():
                metrics[f"explor/{dom}/{m.upper()}/{k[7:]}_{tag}" if k.startswith("explor_") else f"{dom}/{m.upper()}/{k}_{tag}"] = v
    return rows


def explor_indep(coh, probs, target, folds, keep_cal, metrics):
    """Worker-bootstrap view (A1 of the revision) of the ECE after worker-separated calibration, on the pooled held-out confidences of
    all folds (NOT the fold mean of the criterion), for the pre-specified grid and the post-hoc wide grid, plus the uncalibrated
    confidence on the same evaluated rows. Writes explor_boot.csv, explor_per_worker.csv and explor_cal_rows.csv (one row per image per
    fold: calibrated confidence and correctness, because an image evaluated in several manufacturing folds has no single value)."""
    import exp_r2_empirical_noise as E     # lazy: E imports this module at its top (circular otherwise)
    wk = np.array([f"{a}_{b}" for a, b in zip(coh.factory_id, coh.worker_id)])
    stats = lambda conf, corr: np.array([E.ece_cc(conf, corr), corr.mean(), conf.mean()])  # noqa: E731
    boot, perw, long = [], [], []
    for (dom, m, grid), res in keep_cal.items():
        vals = [np.array(va) for _, va in folds[dom]]
        sets = [(f"indep_cal_{grid}", [x[3] for x in res], [x[4] for x in res])]
        if grid == "def":       # uncalibrated reference, once per cell
            raw = probs[m].max(1)
            ok = (np.argmax(probs[m], 1) + METHODS[m][0] == target[m]).astype(float)
            sets.append(("indep_raw", [raw[v] for v in vals], [ok[v] for v in vals]))
        for name, cf, co in sets:
            b, w = E.explor_block(dict(analysis=name, domain=dom, method=m.upper(), target="grand"), dom, vals, wk,
                                  dict(conf=cf, corr=co), stats, ["ece", "acc", "mean_conf"])
            boot += b
            perw += w
        for k, (x, v) in enumerate(zip(res, vals)):
            long += [dict(domain=dom, method=m.upper(), grid=grid, fold=k, T=x[0], file_stem=coh.file_stem[i], worker=wk[i],
                          conf_cal=c, correct=o) for i, c, o in zip(v, x[3], x[4])]
    pd.DataFrame(boot).to_csv(OUT / "explor_boot.csv", index=False)
    pd.DataFrame(perw).to_csv(OUT / "explor_per_worker.csv", index=False)
    pd.DataFrame(long).to_csv(OUT / "explor_cal_rows.csv", index=False)
    E.put_metrics(metrics, boot, ("analysis", "domain", "method", "target", "basis", "stat"))


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"manufacturing": "#1b6ca8", "eldercare": "#c0392b", "logistics": "#2e8b57"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for a, m in zip(ax, METHODS):
        a.plot([0, 1], [0, 1], color="0.6", lw=1)
        for dom in DOMAINS:
            raw, cal = curves[(dom, m)]
            a.plot(*zip(*[(c, k) for c, k, _ in raw]), "o-", color=col[dom], label=f"{dom} raw (T=1)")
            a.plot(*zip(*[(c, k) for c, k, _ in cal]), "s--", color=col[dom], mfc="none", label=f"{dom} worker-held-out T")
        a.set_title(f"{m.upper()} vs annotation-derived score")
        a.set_xlabel("top-class confidence")
        a.set_xlim(0, 1.02)
        a.set_ylim(0, 1.02)
    ax[0].set_ylabel("accuracy against annotation-derived score")
    ax[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = build_cohort()
    n_ok = int(((pd.read_csv(BATCH / "summary_with_mc.csv").status == "ok")).sum())
    print(f"PWW_PROGRESS 0/4 cohort n={len(coh)} of {n_ok}", flush=True)
    # the cohort must equal the one eval_action_agreement scored (5,889 images)
    ref = pd.read_csv(ROOT / "topic/E/result/action_agreement/per_image_labels.csv")
    assert set(ref.file_stem) == set(coh.file_stem), "cohort differs from per_image_labels.csv"
    smoke = int(os.environ.get("SMOKE", "0"))        # smoke test: random subsample of this many images
    if smoke:
        coh = coh.sample(smoke, random_state=0).reset_index(drop=True)

    with Pool(12) as pool:
        tgt = dict(pool.map(_target_worker, [(r["file_stem"], r["process_label"], r) for r in coh.to_dict("records")], chunksize=20))
        print("PWW_PROGRESS 1/4 targets", flush=True)
        mm = dict(pool.map(_mc_matched_worker, [(r.file_stem, r.process_label, r.domain, r.process_id) for r in coh.itertuples()], chunksize=10))
        print("PWW_PROGRESS 2/4 matched MC", flush=True)

    # cached MC (primary) aligned to the cohort order
    z = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
    pos = {s: i for i, s in enumerate(z["file_stem"])}
    ix = [pos[s] for s in coh.file_stem]
    cached = {m: z[f"{m}_probs"][ix] for m in METHODS}
    cached = {m: p / p.sum(1, keepdims=True) for m, p in cached.items()}
    matched = {m: np.array([mm[s][k] for s in coh.file_stem]) for k, m in enumerate(METHODS)}

    # sanity: the cached MC corresponds to current settings (40-image spot check against a fresh run, counted in metrics.json)
    full = [coh.file_stem[i] for i in range(0, len(coh), len(coh) // 40)]
    bad = 0
    for s in full:
        r = coh[coh.file_stem == s].iloc[0]
        res = mc_score_all(EA.system_pose(s, False), config=MC_CFG, world_up=EA.WU, walking=False,
                           **cohort_context(r.process_label), domain=r.domain, process_id=int(r.process_id) or None)
        i = list(coh.file_stem).index(s)
        bad += int(np.abs(res.reba_dist.probs - cached["reba"][i]).max() > 1e-4 or np.abs(res.rula_dist.probs - cached["rula"][i]).max() > 1e-4)
        assert res.baseline.reba.grand_score == r.reba_grand and res.baseline.rula.grand_score == r.rula_grand
        assert res.baseline.owas.action_category == r.owas_action
    metrics = {"n_cohort_ok": n_ok, "n_images": len(coh), "cached_mc_spotcheck_n": len(full), "cached_mc_spotcheck_mismatch": bad}

    ann = {m: np.array([tgt[s][k] for s in coh.file_stem]) for k, m in enumerate(METHODS)}
    det = {m: coh[DET_COL[m]].to_numpy().astype(int) for m in METHODS}
    e12 = _load_exp012()
    folds = make_folds(coh, e12)
    curves = {}
    cal = {}
    ind = evaluate(coh, cached, ann, folds, "indep", metrics, curves, cal)
    explor_indep(coh, cached, ann, folds, cal, metrics)
    slf = evaluate(coh, cached, det, folds, "self", metrics)
    ind_m = evaluate(coh, matched, ann, folds, "indep_matched", metrics)
    print("PWW_PROGRESS 3/4 metrics", flush=True)

    rows = []
    for (dom, m), r in ind.items():
        rows.append({"domain": dom, "method": m.upper(), **{k: v for k, v in r.items()},
                     **{f"self_{k}": slf[(dom, m)][k] for k in ("ece_a", "ece_b", "ece_c_def_mean", "mode_acc")},
                     **{f"matched_{k}": ind_m[(dom, m)][k] for k in ("mode_acc", "coverage", "ece_a", "ece_b", "ece_c_def_mean", "ece_c_ext_mean")},
                     **{f"explor_matched_{k[7:]}": v for k, v in ind_m[(dom, m)].items() if k.startswith("explor_")}})
    pd.DataFrame(rows).to_csv(OUT / "summary_table.csv", index=False)
    metrics["criterion/cells_pass_def"] = int(sum(r["pass_def"] for r in ind.values()))
    metrics["criterion/cells_pass_ext"] = int(sum(r["pass_ext"] for r in ind.values()))
    metrics["criterion/cells_pass_def_matched"] = int(sum(r["pass_def"] for r in ind_m.values()))
    metrics["explor/cells_within_threshold"] = int(sum(r["explor_pass"] for r in ind.values()))
    metrics["explor/cells_within_threshold_matched"] = int(sum(r["explor_pass"] for r in ind_m.values()))
    metrics["criterion/cells_b_worse_than_a"] = int(sum(r["b_worse_than_a"] for r in ind.values()))
    metrics["criterion/b_worse_cells"] = [f"{d}/{m.upper()}" for (d, m), r in ind.items() if r["b_worse_than_a"]]

    per = coh[["file_stem", "domain", "factory_id", "worker_id", "process_label"]].copy()
    for m in METHODS:
        per[f"annot_{m}"] = ann[m]
        per[f"det_{m}"] = det[m]
        per[f"mode_{m}"] = [int(METHODS[m][0] + np.argmax(p)) for p in cached[m]]
        per[f"conf_{m}"] = cached[m].max(1)
        per[f"mode_matched_{m}"] = [int(METHODS[m][0] + np.argmax(p)) for p in matched[m]]
    per.to_csv(OUT / "per_image_indep.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    plot(curves, OUT / "reliability_indep")
    print("PWW_PROGRESS 4/4 done", flush=True)
    print(pd.DataFrame(rows).round(3).T.to_string())


if __name__ == "__main__":
    main()
