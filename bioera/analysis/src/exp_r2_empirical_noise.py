# PWW-DISCLOSURE: public
"""exp-r2-empirical-noise: Monte Carlo with an EMPIRICAL keypoint-error model (A) and conformal sets (B).

Why: exp-r2-indep-calibration showed the Gaussian jitter (sigma = 0.04 x torso x (1 - conf)) is far
too small, light-tailed, uncorrelated across joints and unrelated to the real error. Here the jitter is
a whole-pose RESIDUAL (annotation - prediction, pelvis-origin, own-torso-normalised) resampled from
TRAINING workers of the same domain. Design, metrics and decision rules are fixed in
bioera/analysis/paper/targets/targets.md, section "exp-r2-empirical-noise". Nothing is tuned after seeing results.

No leakage: for every worker-held-out fold the donor workers (residuals, temperature, conformal
threshold) and the evaluated workers are disjoint; asserted inside every task and written to
folds_workers.csv. Calibration images are scored with donors that also exclude their OWN worker, so
the calibration set looks like the test situation (a new worker). Outputs go to $PWW_RESULT_DIR.

Exploratory additions after the internal review (metrics keys 'explor/...', files explor_*.csv; no pass/fail criterion, the
pre-registered metrics are unchanged): worker-level bootstrap (wk_boot, shared by the other exp_r2 scripts; manufacturing only,
eldercare/logistics get per-fold values), A1 vs the sigma 0.04 Gaussian and vs a size-matched Gaussian (sigma_base 0.04-0.20
chosen per fold on the training workers), per-image Brier columns, seed repetition (EXPLOR_SEEDS, default 0,1,2).
"""
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_action_agreement as EA  # noqa: E402
import exp_r2_indep_calibration as X  # noqa: E402  (cohort, folds, thresholds, grid; imported, not copied)
from ace_pose.confidence_mc import McConfig, sample_poses  # noqa: E402
from ace_pose.ergo_v2.context_policy import cohort_context  # noqa: E402
from ace_pose.ergo_v2.pipeline import score_all  # noqa: E402
from ace_pose.major_joints import NECK, PELVIS, Major25Pose  # noqa: E402
from pww_data import resolve  # noqa: E402

EA.BATCH = Path(resolve("cohort_pose_cache"))  # EA.system_pose reads EA.BATCH (X sets the same value)

OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "bioera/analysis/result/exp-r2-empirical-noise"))
DOMAINS, METHODS, THRESH = X.DOMAINS, list(X.METHODS), X.THRESH
J = np.array(sorted(EA.M25_FROM_LABEL))        # 17 joints the annotation has (incl. shoulders/hips)
S_SAMPLES, ALPHA, SEED = 128, 0.10, 0
GRID_T = X.GRID_EXPLOR                          # 0.1 ... 50 (targets.md A3)
# count vector layout: owas action cat | rula grand | reba grand | rula action | reba action
FIELDS = [(0, 1, 4), (4, 1, 7), (11, 1, 15), (26, 1, 4), (30, 0, 5)]   # (offset, first class, n classes)
NCOUNT = 35
SPEC = {("owas", "grand"): 0, ("rula", "grand"): 1, ("reba", "grand"): 2,
        ("owas", "action"): 0, ("rula", "action"): 3, ("reba", "action"): 4}
TARGETS = ("grand", "action")
ACT_OF_GRAND = {"owas": lambda g: g,
                "rula": lambda g: [1, 1, 2, 2, 3, 3, 4][g - 1],
                "reba": lambda g: 0 if g == 1 else 1 if g <= 3 else 2 if g <= 7 else 3 if g <= 10 else 4}
G: dict = {}                                    # read-only globals shared with forked workers


# ---------------------------------------------------------------- deterministic helpers (unit-tested)
def residual(pred_kp, ann_kp, ann_conf):
    """(17, 2) whole-pose residual (annotation - prediction), pelvis origin, each in its own torso length.
    Joints the annotation lacks get 0. None if either torso length is unusable."""
    ref = np.linalg.norm(ann_kp[NECK, :2] - ann_kp[PELVIS, :2])
    rp = np.linalg.norm(pred_kp[NECK, :2] - pred_kp[PELVIS, :2])
    if ann_conf[NECK] <= 0 or ann_conf[PELVIS] <= 0 or ref <= 0 or rp <= 0:
        return None
    a = (pred_kp[J, :2] - pred_kp[PELVIS, :2]) / rp
    b = (ann_kp[J, :2] - ann_kp[PELVIS, :2]) / ref
    return np.where((ann_conf[J] > 0)[:, None], b - a, 0.0)


def apply_residual(pred_kp, res):
    """Add residual (x predicted torso length) to the 17 joints, then rebuild neck/pelvis from the
    perturbed shoulders/hips, as with_neck_pelvis does for the annotation. Depth column untouched."""
    kp = pred_kp.copy()
    rp = np.linalg.norm(pred_kp[NECK, :2] - pred_kp[PELVIS, :2])
    kp[J, :2] += rp * res
    kp[NECK] = (kp[5] + kp[6]) / 2
    kp[PELVIS] = (kp[11] + kp[12]) / 2
    return kp


def greedy_path(p):
    """Nested contiguous sets: start at the mode, add the more probable neighbour each step (tie: lower side).
    Returns lo[k], hi[k] and u[k] = cumulative mass + 1e-9*k (strictly increasing so that classes with
    zero mass can still be reached when tau exceeds 1; keeps the conformal guarantee exact)."""
    K = len(p)
    lo = hi = int(np.argmax(p))
    los, his, cum, us = [lo], [hi], float(p[lo]), [float(p[lo])]
    for k in range(1, K):
        left = p[lo - 1] if lo > 0 else -1.0
        right = p[hi + 1] if hi < K - 1 else -1.0
        if right > left:
            hi += 1
            cum += p[hi]
        else:
            lo -= 1
            cum += p[lo]
        los.append(lo), his.append(hi), us.append(cum + 1e-9 * k)
    return np.array(los), np.array(his), np.array(us)


def set_for_tau(p, tau):
    """Smallest nested set whose u >= tau (whole range if tau is beyond the path)."""
    lo, hi, u = greedy_path(p)
    k = int(np.argmax(u >= tau)) if (u >= tau).any() else len(p) - 1
    return int(lo[k]), int(hi[k])


def score_of(p, y):
    """Conformal score: u of the smallest nested set that contains class index y."""
    lo, hi, u = greedy_path(p)
    return float(u[int(np.argmax((lo <= y) & (y <= hi)))])


def conformal_quantile(scores, alpha):
    """Finite-sample quantile: the ceil((n+1)(1-alpha))-th smallest score; +inf if that rank exceeds n."""
    s = np.sort(np.asarray(scores, float))
    r = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[r - 1]) if r <= len(s) else float("inf")


def softmax_T(P, T):
    """Row-wise temperature scaling, identical to confidence_mc.apply_temperature."""
    lp = np.log(np.clip(P, 1e-12, 1.0)) / max(T, 1e-6)
    lp = lp - lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def fit_T(P, y):
    """Grid temperature minimising NLL (first minimum, as fit_temperature)."""
    nll = [-np.log(softmax_T(P, T)[np.arange(len(y)), y] + 1e-12).sum() for T in GRID_T]
    return float(GRID_T[int(np.argmin(nll))])


def ece_vec(P, y, n_bins=10):
    """Top-class ECE, same bins as confidence_mc.expected_calibration_error."""
    return ece_cc(P.max(1), (P.argmax(1) == y).astype(float), n_bins)


def ece_cc(conf, corr, n_bins=10):
    """ECE from a confidence and a 0/1 correctness vector (exp_r2_indep_calibration reuses it for calibrated confidences)."""
    edges, e = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi + (1e-9 if hi >= 1 else 0))
        if mk.any():
            e += mk.mean() * abs(corr[mk].mean() - conf[mk].mean())
    return float(e)


def brier_rows(P, y):
    """Per-image multi-class Brier (sum over classes of the squared error)."""
    oh = np.zeros_like(P)
    oh[np.arange(len(y)), y] = 1
    return ((P - oh) ** 2).sum(1)


def brier_vec(P, y):
    return float(brier_rows(P, y).mean())


def cov_5_95(P, y):
    cdf = np.cumsum(P, 1)
    lo = (cdf < 0.05).sum(1)
    hi = np.minimum((cdf < 0.95).sum(1), P.shape[1] - 1)
    return float(((lo <= y) & (y <= hi)).mean())


# ---------------------------------------------------------------- explor helpers (reviewer-driven, exploratory, no pass/fail)
N_BOOT_WK, GAUSS_SIGMAS = 2000, (0.04, 0.08, 0.12, 0.16, 0.20)


def wk_boot(wk, stat, n_boot=N_BOOT_WK, seed=0):
    """Worker-level percentile bootstrap (shared by every exp_r2 script): resample the workers of `wk` (one id per
    evaluated row; a worker's rows may repeat over folds) with replacement, keep all rows of each drawn worker.
    stat(row_index_array) -> scalar or 1-D array. Returns (point on all rows, 2.5th, 97.5th percentile) as 1-D arrays."""
    uw, inv = np.unique(wk, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(uw))]
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(stat(np.arange(len(wk)))).astype(float)
    b = np.array([np.atleast_1d(stat(np.concatenate([rows[k] for k in rng.integers(0, len(uw), len(uw))]))) for _ in range(n_boot)], float)
    return point, np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def mean_by_key(keys, *arrs):
    """Average each array over the rows that share a key (an image evaluated in several folds). Returns (unique keys, [means])."""
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    out = []
    for a in arrs:
        a = np.asarray(a, float)
        t = np.zeros((len(u),) + a.shape[1:])
        np.add.at(t, inv, a)
        out.append(t / n.reshape((-1,) + (1,) * (a.ndim - 1)))
    return u, out


def explor_block(ident, dom, vals, wk_all, arrs, stat, names):
    """One cell of an exploratory statistic. vals = per-fold arrays of evaluated image indices, arrs = {name: [per-fold array]},
    stat(**arrays) -> array of len(names). Returns (boot_rows, worker_rows).
    boot_rows: basis 'rows' (every fold counts, an image repeated over folds counts each time) and 'image_once' (arrays averaged
    over the folds an image appears in, then stat). Manufacturing (37 workers) gets a 95% worker-bootstrap interval; eldercare (4) and
    logistics (6) get NO interval (4-6 workers cannot support a bootstrap) and instead worker_rows: the stat on every fold (= one
    held-out worker) so the reader sees all values."""
    img = np.concatenate(vals)
    wk = wk_all[img]
    A = {k: np.concatenate(v) for k, v in arrs.items()}
    boot = []
    for basis in ("rows", "image_once"):
        if basis == "rows":
            w, B, ni = wk, A, len(set(img))
        else:
            ui, first = np.unique(img, return_index=True)
            w, B, ni = wk[first], dict(zip(A, mean_by_key(img, *A.values())[1])), len(ui)
        f = lambda ix: stat(**{k: v[ix] for k, v in B.items()})  # noqa: E731
        if dom == "manufacturing":
            pt, lo, hi = wk_boot(w, f)
        else:
            pt, lo, hi = np.atleast_1d(f(np.arange(len(w)))).astype(float), np.full(len(names), np.nan), np.full(len(names), np.nan)
        boot += [dict(ident, basis=basis, stat=nm, point=pt[j], ci_lo=lo[j], ci_hi=hi[j], n_rows=len(w), n_images=ni,
                      n_workers=len(set(wk)), n_boot=N_BOOT_WK if dom == "manufacturing" else 0) for j, nm in enumerate(names)]
    per_w = []
    if dom != "manufacturing":
        for k, v in enumerate(vals):
            val = np.atleast_1d(stat(**{n: a[k] for n, a in arrs.items()}))
            per_w.append(dict(ident, fold=k, worker=";".join(sorted(set(wk_all[v]))), n_images=len(v), **dict(zip(names, val))))
    return boot, per_w


def put_metrics(metrics, rows, keys):
    """Flatten explor_block rows into metrics under 'explor/<keys...>/<point|ci_lo|ci_hi>' (NaN intervals are left out)."""
    for r in rows:
        for fld in ("point", "ci_lo", "ci_hi"):
            if not pd.isna(r[fld]):
                metrics["explor/" + "/".join(str(r[k]) for k in keys) + f"/{fld}"] = float(r[fld])


def ab_stat(P1, Pg, y):
    """A1 vs a comparison distribution: ECE and Brier of each and their differences (A1 minus comparison)."""
    y = np.asarray(y).astype(int)                # image_once averages y (identical values) into floats
    e1, eg, b1, bg = ece_vec(P1, y), ece_vec(Pg, y), brier_vec(P1, y), brier_vec(Pg, y)
    return np.array([e1, eg, e1 - eg, b1, bg, b1 - bg])


AB_NAMES = ["ece_A1", "ece_cmp", "ece_A1_minus_cmp", "brier_A1", "brier_cmp", "brier_A1_minus_cmp"]


# ---------------------------------------------------------------- scoring workers
def _counts(poses, i):
    """Score poses with the same call as mc_score_all and histogram the five outputs."""
    ctx = G["CTX"][i]
    c = np.zeros(NCOUNT, np.int16)
    for p in poses:
        try:
            r = score_all(p, world_up=EA.WU, walking=False, bilateral=True, domain=G["DOM"][i], process_id=G["PID"][i], **ctx)
        except Exception:
            continue
        for (off, lo, n), v in zip(FIELDS, (r.owas.action_category, r.rula.grand_score, r.reba.grand_score,
                                            r.rula.action_level, r.reba.action_level)):
            if lo <= v < lo + n:
                c[off + v - lo] += 1
    return c


def _w_gauss_cfg(task):
    """Gaussian jitter on the joint-restricted predicted pose; task = (image, sigma_base, seed). S stays X.MC_CFG.n_samples."""
    i, sig, seed = task
    return i, _counts(sample_poses(G["POSE"][i], McConfig(n_samples=X.MC_CFG.n_samples, sigma_fraction=sig, rng_seed=seed)), i)


def _w_gauss(i):
    """Baseline: the current Gaussian jitter (sigma 0.04, seed 0 = X.MC_CFG, same as the old MC)."""
    return _w_gauss_cfg((i, X.MC_CFG.sigma_fraction, X.MC_CFG.rng_seed))


def _w_zero(i):
    """Sanity (a): zero residuals -> every sample equals the deterministic matched-joint pose."""
    kp = apply_residual(G["KP"][i], np.zeros((len(J), 2)))
    return i, _counts([_pose(kp, i)] * 2, i)


def _pose(kp, i):
    return Major25Pose(keypoints=kp, confidence=G["CF"][i], source=np.zeros(25, np.int8))


def _w_emp(task):
    """mode 0 = A1 eval, 1 = A2 eval (donors in the same mean-confidence tercile), 2 = A1 calibration image
    (donors exclude its own worker). Whole residual vectors are drawn with replacement, S = 128."""
    g, i, mode = task[:3]
    seed = task[3] if len(task) > 3 else SEED          # optional 4th element: exploratory seed repetition
    f = G["FOLD"][g]
    D = f["donors"]
    if mode == 2:
        assert not f["isval"][i], "calibration image from an evaluated worker"
        D = D[G["WK"][D] != G["WK"][i]]
    else:
        assert f["isval"][i], "evaluated image not in a held-out worker"
    if mode == 1:
        cut = f["cuts"]
        D = D[np.searchsorted(cut, G["MC"][D]) == np.searchsorted(cut, G["MC"][i])]
    fallback = 0
    if len(D) == 0:
        D, fallback = f["donors"], 1
    assert not f["isval"][D].any(), "LEAK: donor from an evaluated worker"
    if mode == 2:
        assert (G["WK"][D] != G["WK"][i]).all()
    rng = np.random.default_rng([seed, g, i, mode])
    res = G["R"][D[rng.integers(0, len(D), S_SAMPLES)]]
    poses = [_pose(apply_residual(G["KP"][i], r), i) for r in res]
    return (g, i, mode), _counts(poses, i), len(D), fallback


def run_extra_mc(pool, N, folds):
    """Extra MC passes for the exploratory analyses; the main analysis (sigma 0.04, seed 0) is not touched.
    gs[sigma] = Gaussian counts at the other sigma_base values (same joints, S, seed 0); sd[seed] = (gauss counts, {(fold, image, 0): A1 counts})
    for the other seeds in X.EXPLOR_SEEDS. A1 needs only the evaluation images (mode 0)."""
    gs, sd = {}, {}
    for sig in GAUSS_SIGMAS[1:]:
        gs[sig] = np.zeros((N, NCOUNT), np.int16)
        for i, c in pool.imap_unordered(_w_gauss_cfg, [(i, sig, 0) for i in range(N)], chunksize=20):
            gs[sig][i] = c
    for sd_ in [s for s in X.EXPLOR_SEEDS if s != 0]:
        g = np.zeros((N, NCOUNT), np.int16)
        for i, c in pool.imap_unordered(_w_gauss_cfg, [(i, X.MC_CFG.sigma_fraction, sd_) for i in range(N)], chunksize=20):
            g[i] = c
        sd[sd_] = (g, {k: c for k, c, _, _ in pool.imap_unordered(
            _w_emp, [(f["g"], int(i), 0, sd_) for f in folds for i in f["val"]], chunksize=40)})
    return gs, sd


def explor(bycell, ann, gauss, gs, sd, metrics):
    """Exploratory additions (reviewer-driven, post hoc, no pass/fail): A1 worker bootstrap vs sigma 0.04 gauss and vs a size-matched
    gauss, sigma sweep, seed repetition. Writes explor_*.csv and metrics keys 'explor/...'."""
    gs = {GAUSS_SIGMAS[0]: gauss, **gs}
    wk_all = G["WK"]
    boot, perw, sweep, sized_rows, seeds = [], [], [], [], []
    for (dom, m, tg), d in bycell.items():
        field = SPEC[(m, tg)]
        fl, ys, tv = d["fl"], d["ys"], ann[:, field]
        ident = dict(domain=dom, method=m.upper(), target=tg)
        # size-matched gauss: per fold the sigma with the smallest Brier on the TRAINING workers (gauss needs no donors, so no leakage)
        sized, pick = [], []
        for f in fl:
            ytr = y_index(tv[f["train"]], field)
            br = [brier_vec(probs_of(gs[sg][f["train"]], field), ytr) for sg in GAUSS_SIGMAS]
            k = int(np.argmin(br))
            sized.append(probs_of(gs[GAUSS_SIGMAS[k]][f["val"]], field))
            pick.append(GAUSS_SIGMAS[k])
            sized_rows.append(dict(ident, fold=f["g"], sigma_base=GAUSS_SIGMAS[k], train_brier=br[k]))
        for cmp, Pc in (("gauss0.04", d["Ps"]["gauss"]), ("sized_gauss", sized)):
            b, w = explor_block(dict(ident, analysis=f"A1_vs_{cmp}"), dom, [f["val"] for f in fl], wk_all,
                                dict(P1=d["Ps"]["A1"], Pg=Pc, y=ys), ab_stat, AB_NAMES)
            boot += b
            perw += w
        variants = {"A1": d["Ps"]["A1"], "sized_gauss": sized, **{f"gauss{sg:g}": [probs_of(gs[sg][f["val"]], field) for f in fl] for sg in GAUSS_SIGMAS}}
        for v, Ps in variants.items():
            r = cell(Ps, ys, THRESH[m])
            # ece_fold_mean: 사전 기준의 판정 집계(fold 평균)와 같은 집계를 변형마다 남긴다(재심 지적: pooled 와 섞어 비교하지 않게)
            sweep.append(dict(variant=v, **ident, **{k: r[k] for k in ("n", "mode_acc", "mean_top_conf", "ece", "brier", "ece_fold_min", "ece_fold_max")},
                              ece_fold_mean=float(np.mean([ece_vec(p, t) for p, t in zip(Ps, ys)])),
                              brier_fold_mean=float(np.mean([brier_vec(p, t) for p, t in zip(Ps, ys)]))))
        for sdn in X.EXPLOR_SEEDS:
            for v in ("A1", "gauss"):
                if sdn == 0:
                    Ps = d["Ps"][v]
                elif v == "gauss":
                    Ps = [probs_of(sd[sdn][0][f["val"]], field) for f in fl]
                else:
                    Ps = [probs_of(np.array([sd[sdn][1][(f["g"], int(i), 0)] for i in f["val"]]), field) for f in fl]
                P, y = np.concatenate(Ps), np.concatenate(ys)
                seeds.append(dict(seed=sdn, variant=v, **ident, ece=ece_vec(P, y), brier=brier_vec(P, y)))
    pd.DataFrame(boot).to_csv(OUT / "explor_boot.csv", index=False)
    pd.DataFrame(perw).to_csv(OUT / "explor_per_worker.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "explor_sigma_sweep.csv", index=False)
    pd.DataFrame(sized_rows).to_csv(OUT / "explor_sized_sigma.csv", index=False)
    SD = pd.DataFrame(seeds)
    SD.to_csv(OUT / "explor_seeds.csv", index=False)
    put_metrics(metrics, boot, ("analysis", "domain", "method", "target", "basis", "stat"))
    for r in sweep:
        for k in ("ece", "brier"):
            metrics[f"explor/sigma/{r['variant']}/{r['domain']}/{r['method']}/{r['target']}/{k}"] = r[k]
    # std over seeds (ddof=1; NaN with a single seed); every seed value is also kept
    for (v, dom, m, tg), g in SD.groupby(["variant", "domain", "method", "target"]):
        for k in ("ece", "brier"):
            metrics[f"explor/seeds/{v}/{dom}/{m}/{tg}/{k}_std"] = float(g[k].std(ddof=1)) if len(g) > 1 else float("nan")
            for sdn, x in zip(g.seed, g[k]):
                metrics[f"explor/seeds/{v}/{dom}/{m}/{tg}/{k}_seed{sdn}"] = float(x)
    metrics["explor/n_boot_workers"], metrics["explor/seeds"] = N_BOOT_WK, list(X.EXPLOR_SEEDS)


# ---------------------------------------------------------------- data preparation
def prepare(coh):
    """Per-image arrays: predicted pose (matched joints), annotation targets, residuals, deterministic scores."""
    N = len(coh)
    recs = coh.to_dict("records")
    G.update(KP=[], CF=[], POSE=[], R=np.zeros((N, len(J), 2), np.float32), VALID=np.zeros(N, bool),
             MC=np.zeros(N), CTX=[], DOM=list(coh.domain), PID=[int(p) or None for p in coh.process_id],
             WK=np.array([f"{a}_{b}" for a, b in zip(coh.factory_id, coh.worker_id)]))
    ann, det = np.zeros((N, 5), int), np.zeros((N, 5), int)
    for i, r in enumerate(recs):
        p = EA.system_pose(r["file_stem"], True)
        G["KP"].append(p.keypoints), G["CF"].append(p.confidence), G["POSE"].append(p)
        G["CTX"].append(cohort_context(r["process_label"]))
        G["MC"][i] = p.confidence[J].mean()
        lp = EA.label_pose(r)
        res = residual(p.keypoints, lp.keypoints, lp.confidence)
        if res is not None:
            G["R"][i], G["VALID"][i] = res, True
        for arr, pose, extra in ((ann, lp, {}), (det, p, {"domain": r["domain"], "process_id": G["PID"][i]})):
            s = score_all(pose, world_up=EA.WU, walking=False, bilateral=True, **G["CTX"][i], **extra)
            arr[i] = (s.owas.action_category, s.rula.grand_score, s.reba.grand_score, s.rula.action_level, s.reba.action_level)
    G["KP"], G["CF"] = np.array(G["KP"]), np.array(G["CF"])
    return ann, det


def build_folds(coh, e12):
    """Same worker folds as exp-r2-indep-calibration; adds donor pools, tercile cut points, held-out masks."""
    folds, rows = [], []
    for dom, fl in X.make_folds(coh, e12).items():
        for k, (tr, va) in enumerate(fl):
            g = len(folds)
            tr, va = np.array(tr), np.array(va)
            isval = np.zeros(len(coh), bool)
            isval[va] = True
            assert not (set(G["WK"][tr]) & set(G["WK"][va])), "train/val workers overlap"
            donors = tr[G["VALID"][tr]]
            folds.append(dict(g=g, domain=dom, k=k, train=tr, val=va, donors=donors, isval=isval,
                              cuts=np.quantile(G["MC"][donors], [1 / 3, 2 / 3])))
            for role, ix in (("train_donor", tr), ("evaluated", va)):
                for w in sorted(set(G["WK"][ix])):
                    rows.append(dict(fold=g, domain=dom, fold_in_domain=k, role=role, worker=w, n_images=int((G["WK"][ix] == w).sum())))
    G["FOLD"] = {f["g"]: f for f in folds}
    return folds, pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics
def probs_of(counts, field):
    off, _, n = FIELDS[field]
    c = counts[:, off:off + n].astype(float)
    s = c.sum(1, keepdims=True)
    return np.where(s > 0, c / np.maximum(s, 1), 1.0 / n)


def y_index(vals, field):
    return (vals - FIELDS[field][1]).astype(int)


def cell(Ps, ys, thr, base_brier=None):
    """Pooled-over-folds metrics + fold range for one list of per-fold (P, y)."""
    P, y = np.concatenate(Ps), np.concatenate(ys)
    ec = [ece_vec(p, t) for p, t in zip(Ps, ys)]
    r = dict(n=len(y), mode_acc=float((P.argmax(1) == y).mean()), mean_top_conf=float(P.max(1).mean()),
             ece=ece_vec(P, y), brier=brier_vec(P, y), coverage_5_95=cov_5_95(P, y),
             ece_fold_min=float(min(ec)), ece_fold_max=float(max(ec)), n_folds=len(Ps), threshold=thr)
    r["pass"] = bool(r["ece"] <= thr)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = int(os.environ.get("SMOKE", "0"))
    coh = X.build_cohort()
    assert len(coh) == 5889, len(coh)
    ref = pd.read_csv(ROOT / "bioera/analysis/result/action_agreement/per_image_labels.csv")
    assert set(ref.file_stem) == set(coh.file_stem), "cohort differs from per_image_labels.csv"
    if smoke:
        coh = coh.sample(smoke, random_state=0).reset_index(drop=True)
    N = len(coh)
    metrics = {"n_images": N, "smoke": smoke}
    ann, det = prepare(coh)
    metrics["n_valid_residual"] = int(G["VALID"].sum())
    metrics["n_pred_torso_zero_or_annot_missing"] = int((~G["VALID"]).sum())
    for arr in (ann, det):
        for k, (_, lo, n) in enumerate(FIELDS):
            assert arr[:, k].min() >= lo and arr[:, k].max() < lo + n, f"score outside class range, field {k}"
    # grand -> action map must match the scorer (used to count sets spanning several action levels)
    for m, (gf, af) in {"rula": (1, 3), "reba": (2, 4), "owas": (0, 0)}.items():
        for arr in (ann, det):
            assert all(ACT_OF_GRAND[m](g) == a for g, a in zip(arr[:, gf], arr[:, af])), f"grand->action map wrong for {m}"
    e12 = X._load_exp012()
    folds, fw = build_folds(coh, e12)
    fw.to_csv(OUT / "folds_workers.csv", index=False)
    for f in folds:     # leakage check, written to metrics
        assert not (set(G["WK"][f["donors"]]) & set(G["WK"][f["val"]]))
    metrics["leakage_check/folds"] = len(folds)
    metrics["leakage_check/donor_eval_worker_overlap"] = 0
    print(f"PWW_PROGRESS 0/6 prepared N={N} folds={len(folds)}", flush=True)

    ctx = get_context("fork")
    with ctx.Pool(12) as pool:
        # sanity (a): zero residual == deterministic matched score
        z = dict(pool.imap_unordered(_w_zero, range(N), chunksize=20))
        ok = 0
        for i in range(N):
            c = z[i]
            md = [FIELDS[k][1] + int(np.argmax(c[FIELDS[k][0]:FIELDS[k][0] + FIELDS[k][2]])) for k in range(5)]
            pure = all(c[FIELDS[k][0]:FIELDS[k][0] + FIELDS[k][2]].max() == 2 for k in range(5))
            ok += int(md == list(det[i]) and pure)
        metrics["sanity_a/zero_residual_equals_deterministic_n"] = ok
        metrics["sanity_a/zero_residual_equals_deterministic_share"] = ok / N
        print(f"PWW_PROGRESS 1/6 sanity-a {ok}/{N}", flush=True)
        gauss = np.zeros((N, NCOUNT), np.int16)
        for i, c in pool.imap_unordered(_w_gauss, range(N), chunksize=20):
            gauss[i] = c
        print("PWW_PROGRESS 2/6 gauss MC", flush=True)
        tasks = [(f["g"], int(i), mode) for f in folds for mode, ix in ((0, f["val"]), (1, f["val"]), (2, f["train"])) for i in ix]
        emp, fb = {}, 0
        for n, (key, c, nd, fbk) in enumerate(pool.imap_unordered(_w_emp, tasks, chunksize=40)):
            emp[key], fb = c, fb + fbk
            if n % 5000 == 0:
                print(f"PWW_PROGRESS {n}/{len(tasks)} empirical MC", flush=True)
        gs, sd = run_extra_mc(pool, N, folds)
    metrics["empirical_tasks"], metrics["donor_pool_empty_fallbacks"] = len(tasks), fb
    print("PWW_PROGRESS 3/6 empirical MC", flush=True)

    # ------------------------------------------------ A: per cell, per variant
    def fold_data(variant, f, field):
        """(val probs, val y, [train probs for calibration])."""
        va, tr = f["val"], f["train"]
        src = (lambda ix, md: gauss[ix]) if variant == "gauss" else (lambda ix, md: np.array([emp[(f["g"], int(i), md)] for i in ix]))
        return src(va, {"gauss": 0, "A1": 0, "A2": 1}[variant]), src(tr, 2) if variant == "A1" else None

    rowsA, bycell, metrics_cells = [], {}, {}
    for dom in DOMAINS:
        fl = [f for f in folds if f["domain"] == dom]
        for m in METHODS:
            for tg in TARGETS:
                field = SPEC[(m, tg)]
                tvals = ann[:, [0, 1, 2, 3, 4][field]]
                Ps = {v: [] for v in ("gauss", "A1", "A2", "A3")}
                ys, Pcal = [], []
                for f in fl:
                    ys.append(y_index(tvals[f["val"]], field))
                    for v in ("gauss", "A1", "A2"):
                        cv, cc = fold_data(v, f, field)
                        Ps[v].append(probs_of(cv, field))
                        if v == "A1":
                            Pcal.append((probs_of(cc, field), y_index(tvals[f["train"]], field)))
                    T = fit_T(*Pcal[-1])
                    Ps["A3"].append(softmax_T(Ps["A1"][-1], T))
                    Ps.setdefault("T", []).append(T)
                bycell[(dom, m, tg)] = dict(Ps=Ps, ys=ys, Pcal=Pcal, fl=fl)
                for v in ("gauss", "A1", "A2", "A3"):
                    r = cell(Ps[v], ys, THRESH[m])
                    r.update(variant=v, domain=dom, method=m.upper(), target=tg)
                    if v == "A3":
                        r["T_mean"], r["T_min"], r["T_max"] = float(np.mean(Ps["T"])), float(min(Ps["T"])), float(max(Ps["T"]))
                    rowsA.append(r)
    A = pd.DataFrame(rowsA)
    gb = A[A.variant == "gauss"].set_index(["domain", "method", "target"]).brier
    A["brier_gauss"] = [gb[(d, m, t)] for d, m, t in zip(A.domain, A.method, A.target)]
    A["brier_below_gauss"] = A.brier < A.brier_gauss
    A = A[["variant", "domain", "method", "target", "n", "mode_acc", "mean_top_conf", "ece", "brier", "coverage_5_95",
           "ece_fold_min", "ece_fold_max", "n_folds", "threshold", "pass", "brier_gauss", "brier_below_gauss",
           "T_mean", "T_min", "T_max"]]
    A.to_csv(OUT / "summary_A.csv", index=False)
    for r in A.to_dict("records"):
        pre = f"{r['variant']}/{r['domain']}/{r['method']}" + ("" if r["target"] == "grand" else "/action")
        for k in ("n", "mode_acc", "mean_top_conf", "ece", "brier", "coverage_5_95", "ece_fold_min", "ece_fold_max", "pass"):
            metrics[f"{pre}/{k}"] = r[k]
    for tg in TARGETS:
        for v in ("A1", "A2", "A3"):
            s = A[(A.variant == v) & (A.target == tg)]
            metrics[f"criterionA/{tg}/{v}/cells_ece_pass"] = int(s["pass"].sum())
            metrics[f"criterionA/{tg}/{v}/cells_brier_below_gauss"] = int(s.brier_below_gauss.sum())
            metrics[f"criterionA/{tg}/{v}/meets"] = bool(s["pass"].sum() >= 7 and s.brier_below_gauss.sum() >= 7)
    g = metrics
    metrics["criterionA/verdict"] = ("A1: calibrated vs annotated keypoints" if g["criterionA/grand/A1/meets"] else
                                     "A3: calibrated after temperature" if g["criterionA/grand/A3/meets"] else "FAIL")
    print("PWW_PROGRESS 4/6 A table", flush=True)

    # sanity (b): gauss whole-domain numbers vs the matched_* columns of exp-r2-indep-calibration
    old = pd.read_csv(ROOT / "bioera/analysis/result/exp-r2-indep-calibration/summary_table.csv").set_index(["domain", "method"])
    dmax, rowsb = {"mode_acc": 0.0, "ece": 0.0}, []
    for dom in DOMAINS:
        ix = np.where(np.array(G["DOM"]) == dom)[0]
        for m in METHODS:
            P, y = probs_of(gauss[ix], SPEC[(m, "grand")]), y_index(ann[ix, SPEC[(m, "grand")]], SPEC[(m, "grand")])
            o = old.loc[(dom, m.upper())]
            d1, d2 = abs((P.argmax(1) == y).mean() - o.matched_mode_acc), abs(ece_vec(P, y) - o.matched_ece_a)
            rowsb.append(dict(domain=dom, method=m.upper(), d_mode_acc=d1, d_ece=d2))
            dmax["mode_acc"], dmax["ece"] = max(dmax["mode_acc"], d1), max(dmax["ece"], d2)
    pd.DataFrame(rowsb).to_csv(OUT / "sanity_b_gauss_vs_old_matched.csv", index=False)
    metrics["sanity_b/max_abs_diff_mode_acc"], metrics["sanity_b/max_abs_diff_ece"] = dmax["mode_acc"], dmax["ece"]
    if smoke:
        metrics["sanity_b/note"] = "subsample: differences vs the full-cohort old numbers are not meaningful"

    # ------------------------------------------------ B: conformal sets
    rowsB, per_rows = [], []
    for (dom, m, tg), d in bycell.items():
        field = SPEC[(m, tg)]
        lo_cls = FIELDS[field][1]
        K = FIELDS[field][2]
        detv = y_index(det[:, field], field)
        out = {}
        for dist in ("A1", "gauss", "fixed_width"):
            cov, size, sing, singacc, multi, tq, cnt = [], [], [], [], [], [], 0
            for f, y, Pv, (Pc, yc) in zip(d["fl"], d["ys"], d["Ps"]["A1" if dist != "gauss" else "gauss"], d["Pcal"]):
                if dist == "gauss":
                    Pc = probs_of(gauss[f["train"]], field)
                if dist == "fixed_width":
                    q = conformal_quantile(np.abs(detv[f["train"]] - yc), ALPHA)
                    q = K if np.isinf(q) else int(q)
                    lo = np.clip(detv[f["val"]] - q, 0, K - 1)
                    hi = np.clip(detv[f["val"]] + q, 0, K - 1)
                    tq.append(q)
                else:
                    tau = conformal_quantile([score_of(p, t) for p, t in zip(Pc, yc)], ALPHA)
                    lh = np.array([set_for_tau(p, tau) for p in Pv])
                    lo, hi = lh[:, 0], lh[:, 1]
                    tq.append(tau)
                cv = (lo <= y) & (y <= hi)
                one = lo == hi
                span = np.array([ACT_OF_GRAND[m](int(a) + lo_cls) != ACT_OF_GRAND[m](int(b) + lo_cls) for a, b in zip(lo, hi)]) \
                    if tg == "grand" else ~one
                cov.append(cv), size.append(hi - lo + 1), sing.append(one), singacc.append(cv[one]), multi.append(span)
                if dist == "A1" or dist == "fixed_width":
                    for j, i in enumerate(f["val"]):
                        per_rows.append(dict(dist=dist, fold=f["g"], domain=dom, method=m, target=tg, file_stem=coh.file_stem[i],
                                             worker=G["WK"][i], target_class=int(y[j]) + lo_cls, set_lo=int(lo[j]) + lo_cls, set_hi=int(hi[j]) + lo_cls))
            C, S1, O = np.concatenate(cov), np.concatenate(size), np.concatenate(sing)
            r = dict(distribution=dist, domain=dom, method=m.upper(), target=tg, n=len(C), coverage=float(C.mean()),
                     coverage_fold_min=float(min(c.mean() for c in cov)), coverage_fold_max=float(max(c.mean() for c in cov)),
                     mean_set_size=float(S1.mean()), singleton_share=float(O.mean()),
                     singleton_accuracy=float(np.concatenate(singacc).mean()) if O.any() else float("nan"),
                     multi_action_level_share=float(np.concatenate(multi).mean()),
                     tau_or_q_per_fold=";".join(f"{t:.4f}" for t in tq), n_folds=len(cov))
            out[dist] = r
            rowsB.append(r)
        out["A1"]["smaller_than_fixed"] = bool(out["A1"]["mean_set_size"] < out["fixed_width"]["mean_set_size"])
    B = pd.DataFrame(rowsB)
    B.to_csv(OUT / "summary_B.csv", index=False)
    for r in rowsB:
        pre = f"B/{r['distribution']}/{r['domain']}/{r['method']}" + ("" if r["target"] == "action" else "/grand")
        for k in ("n", "coverage", "coverage_fold_min", "coverage_fold_max", "mean_set_size", "singleton_share",
                  "singleton_accuracy", "multi_action_level_share"):
            metrics[f"{pre}/{k}"] = r[k]
    for tg in TARGETS:
        s = B[(B.distribution == "A1") & (B.target == tg)]
        metrics[f"criterionB/{tg}/cells_coverage_ge_0.88"] = int((s.coverage >= 0.88).sum())
        metrics[f"criterionB/{tg}/cells_smaller_than_fixed"] = int(s.smaller_than_fixed.sum())
        metrics[f"criterionB/{tg}/verdict"] = ("valid and useful" if (s.coverage >= 0.88).all() and s.smaller_than_fixed.sum() >= 7 else
                                               "valid but not better than fixed width" if (s.coverage >= 0.88).all() else "FAIL")
    metrics["criterionB/verdict"] = metrics["criterionB/action/verdict"]
    pd.DataFrame(per_rows).to_csv(OUT / "per_image_sets.csv", index=False)
    print("PWW_PROGRESS 5/6 B table", flush=True)

    # ------------------------------------------------ per-image distribution summaries (A1, gauss), risk-coverage, plots
    pi = []
    for (dom, m, tg), d in bycell.items():
        field = SPEC[(m, tg)]
        lo_cls = FIELDS[field][1]
        for f, y, Pg, P1 in zip(d["fl"], d["ys"], d["Ps"]["gauss"], d["Ps"]["A1"]):
            bg, b1 = brier_rows(Pg, y), brier_rows(P1, y)
            for j, i in enumerate(f["val"]):
                pi.append(dict(fold=f["g"], domain=dom, method=m, target=tg, file_stem=coh.file_stem[i], worker=G["WK"][i],
                               target_class=int(y[j]) + lo_cls, det_matched=int(det[i, field]),
                               mode_gauss=int(Pg[j].argmax()) + lo_cls, top_gauss=float(Pg[j].max()),
                               mode_A1=int(P1[j].argmax()) + lo_cls, top_A1=float(P1[j].max()),
                               explor_brier_gauss=float(bg[j]), explor_brier_A1=float(b1[j])))
    pd.DataFrame(pi).to_csv(OUT / "per_image.csv", index=False)

    rc, rcrows = {}, []
    grid = np.linspace(0.01, 1, 100)
    for (dom, m, tg), d in bycell.items():
        P, y = np.concatenate(d["Ps"]["A1"]), np.concatenate(d["ys"])
        order = np.argsort(-P.max(1), kind="stable")
        err = (P.argmax(1) != y)[order].astype(float)
        risk = np.cumsum(err) / np.arange(1, len(err) + 1)
        aurc, rand = float(risk.mean()), float(err.mean())
        rc[(dom, m, tg)] = (grid, risk[np.maximum((grid * len(err)).astype(int) - 1, 0)], rand)
        rcrows += [dict(domain=dom, method=m.upper(), target=tg, coverage=c, risk=r_, random_risk=rand) for c, r_ in zip(*rc[(dom, m, tg)][:2])]
        pre = f"riskcov/{dom}/{m.upper()}" + ("" if tg == "grand" else "/action")
        metrics[f"{pre}/area_A1"], metrics[f"{pre}/area_random"] = aurc, rand
    pd.DataFrame(rcrows).to_csv(OUT / "risk_coverage.csv", index=False)
    explor(bycell, ann, gauss, gs, sd, metrics)
    plots(bycell, rc)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("PWW_PROGRESS 6/6 done", flush=True)
    show = A[A.target == "grand"].set_index(["variant", "domain", "method"])[["mode_acc", "mean_top_conf", "ece", "brier", "pass"]]
    print(show.round(3).to_string())


def plots(bycell, rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"manufacturing": "#1b6ca8", "eldercare": "#c0392b", "logistics": "#2e8b57"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for a, m in zip(ax, METHODS):
        a.plot([0, 1], [0, 1], color="0.6", lw=1)
        for dom in DOMAINS:
            d = bycell[(dom, m, "grand")]
            y = np.concatenate(d["ys"])
            for v, st, mf in (("gauss", "o--", "none"), ("A1", "s-", col[dom])):
                bins = X.reliability_bins(np.concatenate(d["Ps"][v]).max(1), (np.concatenate(d["Ps"][v]).argmax(1) == y).astype(float))
                a.plot(*zip(*[(c, k) for c, k, _ in bins]), st, color=col[dom], mfc=mf, label=f"{dom} {v}")
        a.set_title(f"{m.upper()} vs annotation-derived score")
        a.set_xlabel("top-class confidence"), a.set_xlim(0, 1.02), a.set_ylim(0, 1.02)
    ax[0].set_ylabel("accuracy"), ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "reliability_A1.png", dpi=200), fig.savefig(OUT / "reliability_A1.pdf")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, m in zip(ax, METHODS):
        for dom in DOMAINS:
            c, r, rand = rc[(dom, m, "action")]
            a.plot(c, r, color=col[dom], label=f"{dom} A1")
            a.axhline(rand, color=col[dom], ls=":", lw=1)
        a.set_title(f"{m.upper()} (action level; dotted = random order)")
        a.set_xlabel("coverage (kept, most confident first)")
    ax[0].set_ylabel("risk (mode wrong)"), ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=200), fig.savefig(OUT / "risk_coverage.pdf")


if __name__ == "__main__":
    main()
