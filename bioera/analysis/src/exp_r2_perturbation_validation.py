# PWW-DISCLOSURE: public
"""exp-r2-perturbation-validation: descriptive validation of the Monte Carlo keypoint-perturbation model (no pass/fail).

Why: reviewer 5 asked whether the perturbation model and its parameters (sigma_base 0.04, S 128) are validated. The measurements
follow the field's usual elements (COCO OKS per-joint constants, body-normalised position error, joint-angle error in degrees,
Monte Carlo standard error, level retention). Design and interpretation rules are fixed BEFORE running in
bioera/analysis/paper/targets/targets.md, section "섭동 모형 검증 (exp-r2-perturbation-validation ...)". Post hoc analysis (primary and
secondary results were already seen), and sigma_base / S are NOT re-chosen from these data.

Input : cohort of 5,889 images (exp_r2_indep_calibration.build_cohort), predicted-pose cache and annotated keypoints, all read through
        the existing helpers (resolve("cohort_pose_cache"), resolve("aihub_busan_annotations")). Nothing is read by hard-coded path.
Output: $PWW_RESULT_DIR/metrics.json (flat key -> number; every number quoted in the manuscript lives here) and long-form CSVs
        per_joint.csv (items 1-4), angle_deg.csv (5), sample_count.csv (6), output_sensitivity.csv (7).
        Key patterns (all end in /point, /lo, /hi): pj/<domain>/<stat>/<joint>, pj/<domain>/<global>/all,
        angle/<domain>/pert/<angle>/sigma<s>/<median|rms|p95>, angle/<domain>/diff/<angle>/<n|median|mean|p68|p95>,
        count/<domain>/<output>/S<S>/<se_mean|se_p95|disagree|dp_mean|dp_p95>,
        sens/<domain>/<method>/sigma<s>/S<S>/<keep_mode|keep_share|acc_mode>, sens/<domain>/<method>/deterministic/acc_det
Interval: manufacturing = 95% worker-level bootstrap (2,000 draws, seed 0, E.wk_boot); eldercare (4 workers) and logistics (6 workers)
        = min and max over workers (lo/hi); 'all' = point only.

The pipeline's own perturbation (ace_pose/confidence_mc.sample_poses) is reused, not re-implemented: sigma_i = sigma_fraction x
ref_length x (1 - clip(conf_i, 0, 1)), ref_length = max(shoulder width, hip width, torso length) of the predicted pose, Gaussian,
independent per keypoint, on all 25 keypoints and all 3 columns. So in units of ref_length the assumed sd of joint i is
sigma_fraction x (1 - conf_i).

Run: SMOKE=1 for ~200 images. Run time scales with the Monte Carlo (3,072 pipeline scorings per image).
"""
import json
import os
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_r2_empirical_noise as E  # noqa: E402  (prepare, residual, FIELDS, wk_boot, _pose helpers; imported, not copied)
from ace_pose.anthropometry import compute_anthropometry  # noqa: E402
from ace_pose.confidence_mc import DEFAULT_SIGMA_FRACTION, McConfig, sample_poses  # noqa: E402
from ace_pose.ergo_v2 import angles as AN  # noqa: E402
from ace_pose.ergo_v2.pipeline import score_all  # noqa: E402
from ace_pose.major_joints import MAJOR25_NAMES, NECK, PELVIS, Major25Pose  # noqa: E402

EA, X, G = E.EA, E.X, E.G
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN joint columns in tiny per-worker subsets give NaN on purpose
OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "bioera/analysis/result/exp-r2-perturbation-validation"))
DOMAINS = ["all"] + list(X.DOMAINS)
J = E.J                                                  # 17 major-25 indices the annotation has (nose, eyes, 12 body joints, 2 big toes)
JN = [MAJOR25_NAMES[j].lower() for j in J]
# COCO OKS per-joint constants (k_i = 2 sigma_i), 17 values in COCO joint order, from the official COCO evaluation code
# (cocoapi PythonAPI/pycocotools/cocoeval.py, Params.setKpParams: kpt_oks_sigmas). Order: nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
# left_ankle, right_ankle. major-25 indices 0-16 follow the same order, so index j maps to OKS_SIGMAS[j]; the toes (17, 20) have no
# COCO-17 counterpart and are dropped from item 4.
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10
OKS_OF_J = np.array([OKS_SIGMAS[j] if j < 17 else np.nan for j in J])
SIGMAS = (0.01, 0.02, 0.04, 0.08, 0.16)                  # perturbation sizes (targets.md item 5/7); 0.04 = pipeline default
SAMPLE_COUNTS = (32, 64, 128, 256, 512)                  # S values (item 6/7); nested prefixes of one S_MAX draw
S_MAX, S_ANG = 512, 128                                  # S_ANG: samples per image for the angle sd (item 5)
SEEDS_PAIR = (0, 1)                                      # independent seeds for the seed-to-seed comparison
NB = 100 if os.environ.get("SMOKE") == "1" else 2000     # bootstrap draws (manufacturing)
OUTS = [("owas_action", 0), ("rula_grand", 1), ("reba_grand", 2), ("rula_action", 3), ("reba_action", 4)]   # (name, E.FIELDS index)
ACT = [("owas", 0), ("rula", 3), ("reba", 4)]            # action-level fields (item 7)
ANG = ["trunk", "neck", "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "knee_l", "knee_r"]
# major-25 joints that must be annotated before an annotation angle exists (trunk uses neck/pelvis derived from shoulders and hips)
TRUNK_J = [5, 6, 11, 12]
ANG_NEED = [TRUNK_J, TRUNK_J + [1, 2], TRUNK_J + [7], TRUNK_J + [8], [5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16]]
PJ_STATS = ["n", "med", "p68", "p95", "sd_x", "sd_y", "rsd_x", "rsd_y", "sig_med", "sig_rms", "ratio_x", "ratio_y", "ratio",
            "ratio_rms", "kurt_x", "kurt_y", "kurt", "iso", "iso_robust", "rho"]
PJ_GLOBALS = ["ratio_median", "n_ratio_lt05", "n_ratio_05_2", "n_ratio_gt2", "n_kurt_gt1", "n_joints", "rho_abs_median",
              "oks_rho_med", "oks_rho_rsd"]


# ---------------------------------------------------------------- pure helpers (unit-tested)
def normalized_error(res, rp, ref, annotated):
    """Residual (annotation - prediction, in predicted-torso units, E.residual) -> error in pipeline ref_length units.
    res (N, K, 2), rp (N,) predicted 2-D torso length, ref (N,) pipeline ref_length, annotated (N, K) bool. NaN where the joint is not
    annotated or the image has no usable residual (rp or ref <= 0)."""
    out = res * (rp / ref)[:, None, None]
    return np.where(annotated[:, :, None], out, np.nan)


def robust_sd(x):
    """IQR / 1.349 along axis 0, NaN-aware (equals the sd for a normal distribution, insensitive to heavy tails)."""
    q = np.nanpercentile(x, [25, 75], axis=0)
    return (q[1] - q[0]) / 1.349


def excess_kurtosis(x):
    """m4 / m2^2 - 3 along axis 0, NaN-aware (0 for normal, > 0 heavier tails)."""
    c = x - np.nanmean(x, axis=0)
    return np.nanmean(c ** 4, axis=0) / np.nanmean(c ** 2, axis=0) ** 2 - 3


def rank_corr(a, b):
    """Spearman correlation with average ranks for ties, over the pairs where both are finite; NaN if fewer than 3 pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wrapped_std(a, ref):
    """Circular-safe sd of angles a (S, K) in degrees around ref (K,): deviations are folded into [-180, 180) first, so a
    +179 / -179 pair is 2 degrees apart, not 358. NaN-aware."""
    return np.nanstd((a - ref + 180.0) % 360.0 - 180.0, axis=0)


def xy_only(poses, base):
    """Copies of perturbed poses whose z column is put back to the unperturbed pose's z (only x, y stay perturbed)."""
    out = []
    for p in poses:
        kp = p.keypoints.copy()
        kp[:, 2] = base.keypoints[:, 2]
        out.append(Major25Pose(keypoints=kp, confidence=p.confidence, source=p.source))
    return out


def z_range_ratio(kp):
    """Range of the z column over the 25 keypoints / the larger of the x and y ranges (per image)."""
    r = np.ptp(kp, axis=0)
    return float(r[2] / max(r[0], r[1], 1e-12))


def mc_se(pmax, total):
    """Monte Carlo standard error of the modal-class probability: sqrt(p (1 - p) / S)."""
    return np.sqrt(pmax * (1 - pmax) / np.maximum(total, 1))


def class_probs(levels, S, lo, n):
    """Class probabilities from the first S samples. levels (N, S_MAX) int (-1 = failed sample), classes lo..lo+n-1.
    Returns (P (N, n), total valid samples (N,)). A failed sample is not counted (as E._counts)."""
    sub = levels[:, :S]
    c = np.stack([(sub == lo + k).sum(1) for k in range(n)], 1).astype(float)
    tot = c.sum(1)
    return c / np.maximum(tot, 1)[:, None], tot


def col_stats(Q, kinds):
    """Column-wise summary of a per-image matrix Q (rows = images). kinds[k] in mean|median|p5|p68|p95|rms|n, NaN-aware.
    One 1-D vector, so it can be a bootstrap statistic."""
    out = np.full(Q.shape[1], np.nan)
    for kind in set(kinds):
        ix = [k for k, kd in enumerate(kinds) if kd == kind]
        sub = Q[:, ix]
        if kind == "n":
            v = np.isfinite(sub).sum(0)
        elif kind == "mean":
            v = np.nanmean(sub, 0)
        elif kind == "rms":
            v = np.sqrt(np.nanmean(sub ** 2, 0))
        else:
            p = {"median": 50, "p5": 5, "p68": 68, "p95": 95}[kind]
            v = np.nanpercentile(sub, p, axis=0) if np.isnan(sub).any() else np.percentile(sub, p, axis=0)
        out[ix] = v
    return out


def summarize(dom, wk, rows, stat, n_boot=None):
    """(point, lo, hi) of stat(row_indices) -> 1-D vector for one domain.
    manufacturing: worker-level bootstrap 95% interval; eldercare/logistics: min/max over the workers' own values; all: no interval.
    rows = global image indices of the domain, wk = worker id per global image."""
    n_boot = NB if n_boot is None else n_boot
    w = wk[rows]
    if dom == "manufacturing":
        pt, lo, hi = E.wk_boot(w, lambda ix: stat(rows[ix]), n_boot=n_boot)
        return pt, lo, hi
    pt = np.atleast_1d(stat(rows)).astype(float)
    if dom == "all":
        return pt, np.full_like(pt, np.nan), np.full_like(pt, np.nan)
    per = np.array([np.atleast_1d(stat(rows[w == u])) for u in np.unique(w)], float)
    return pt, np.nanmin(per, 0), np.nanmax(per, 0)


# ---------------------------------------------------------------- item 1-4: joint position error vs the assumed perturbation
def pj_names():
    """Layout of the per-joint statistic vector: stat-major over the 17 joints, then the global scalars."""
    return [(s, j) for s in PJ_STATS for j in JN] + [(g, "all") for g in PJ_GLOBALS]


def pj_stat(err, cf, rows):
    """Items 1-4 for the images `rows`. err (N, 17, 2) normalised error (NaN = not annotated), cf (N, 17) predicted confidence.
    ratio = robust sd of the empirical error (both axes combined) / median assumed sd (sigma_fraction x (1 - conf)) of the same joint;
    ratio_rms uses the rms assumed sd. Returns the flat vector in pj_names() order."""
    e, c = err[rows], cf[rows]
    ok = np.isfinite(e[:, :, 0])
    d = np.linalg.norm(e, axis=2)
    sig = np.where(ok, DEFAULT_SIGMA_FRACTION * (1 - np.clip(c, 0, 1)), np.nan)
    q = np.nanpercentile(d, [50, 68, 95], axis=0)
    ex, ey = e[:, :, 0], e[:, :, 1]
    rx, ry = robust_sd(ex), robust_sd(ey)
    s = {"n": ok.sum(0).astype(float), "med": q[0], "p68": q[1], "p95": q[2], "sd_x": np.nanstd(ex, 0), "sd_y": np.nanstd(ey, 0),
         "rsd_x": rx, "rsd_y": ry, "sig_med": np.nanmedian(sig, 0), "sig_rms": np.sqrt(np.nanmean(sig ** 2, 0))}
    comb = np.sqrt((rx ** 2 + ry ** 2) / 2)
    s.update(ratio_x=rx / s["sig_med"], ratio_y=ry / s["sig_med"], ratio=comb / s["sig_med"], ratio_rms=comb / s["sig_rms"],
             kurt_x=excess_kurtosis(ex), kurt_y=excess_kurtosis(ey))
    s["kurt"] = (s["kurt_x"] + s["kurt_y"]) / 2
    s["iso"], s["iso_robust"] = s["sd_x"] / s["sd_y"], rx / ry
    s["rho"] = np.array([rank_corr(c[:, k], d[:, k]) for k in range(len(JN))])
    r = s["ratio"]
    g = {"ratio_median": np.nanmedian(r), "n_ratio_lt05": (r < 0.5).sum(), "n_ratio_05_2": ((r >= 0.5) & (r <= 2)).sum(),
         "n_ratio_gt2": (r > 2).sum(), "n_kurt_gt1": (s["kurt"] > 1).sum(), "n_joints": np.isfinite(r).sum(),
         "rho_abs_median": np.nanmedian(np.abs(s["rho"])), "oks_rho_med": rank_corr(OKS_OF_J, s["med"]),
         "oks_rho_rsd": rank_corr(OKS_OF_J, comb)}
    return np.array([float(s[st][k]) for st in PJ_STATS for k in range(len(JN))] + [float(g[n]) for n in PJ_GLOBALS])


# ---------------------------------------------------------------- item 5: angles
def angles_of(pose):
    """The pipeline's own joint-angle functions (ace_pose.ergo_v2.angles), degrees, in ANG order; NaN if a function fails."""
    try:
        f = AN.build_body_frame(pose)
        return np.array([AN.trunk_flexion_deg(pose, f, world_up=EA.WU), AN.neck_flexion_deg(pose, f, world_up=EA.WU),
                         AN.upper_arm_flexion_deg(pose, f, "left"), AN.upper_arm_flexion_deg(pose, f, "right"),
                         AN.elbow_flexion_deg(pose, "left"), AN.elbow_flexion_deg(pose, "right"),
                         AN.knee_flexion_deg(pose, "left"), AN.knee_flexion_deg(pose, "right")], float)
    except Exception:
        return np.full(len(ANG), np.nan)


def _w_angle_sd(task):
    """Circular sd (degrees) of each angle over S_ANG perturbed copies of the predicted pose, around the unperturbed angle.
    Returns (i, sd with the pipeline's x, y, z perturbation, sd with z restored = only x, y perturbed). Same noise draws for both."""
    i, sig = task
    base = G["POSE"][i]
    poses = sample_poses(base, McConfig(n_samples=S_ANG, sigma_fraction=sig, rng_seed=0))
    ref = angles_of(base)
    return i, wrapped_std(np.array([angles_of(p) for p in poses]), ref), wrapped_std(np.array([angles_of(p) for p in xy_only(poses, base)]), ref)


# ---------------------------------------------------------------- item 6-7: pipeline scoring of the perturbed poses
def _w_levels(task):
    """All S_MAX perturbed copies of image i scored with the same call as E._counts; returns (S_MAX, 5) int8 levels
    (columns = OUTS order; -1 = failed sample). Nested prefixes of this array give every S."""
    i, sig, seed, zfix = task          # zfix: item 8 variant, z restored, only the first S_ANG draws (same draws as the xyz prefix)
    ctx = G["CTX"][i]
    n = S_ANG if zfix else S_MAX
    lev = np.full((n, len(OUTS)), -1, np.int8)
    poses = sample_poses(G["POSE"][i], McConfig(n_samples=n, sigma_fraction=sig, rng_seed=seed))
    for k, p in enumerate(xy_only(poses, G["POSE"][i]) if zfix else poses):
        try:
            r = score_all(p, world_up=EA.WU, walking=False, bilateral=True, domain=G["DOM"][i], process_id=G["PID"][i], **ctx)
        except Exception:
            continue
        lev[k] = (r.owas.action_category, r.rula.grand_score, r.reba.grand_score, r.rula.action_level, r.reba.action_level)
    return task, lev


def sg(s):
    return f"{s:g}"


def record(rows, metrics, prefix, dom, keys, pt, lo, hi, label):
    """Append one long-form CSV row per statistic and the flat metrics keys <prefix>/<domain>/<keys...>/<point|lo|hi>.
    label = how lo/hi were obtained (worker_bootstrap | worker_min_max | none)."""
    for k, p, l, h in zip(keys, pt, lo, hi):
        rows.append(dict(domain=dom, **{f"k{n}": v for n, v in enumerate(k)}, point=p, lo=l, hi=h, interval=label))
        for fld, v in (("point", p), ("lo", l), ("hi", h)):
            if np.isfinite(v):
                metrics["/".join([prefix, dom, *k, fld])] = float(v)


def interval_label(dom):
    return {"manufacturing": "worker_bootstrap", "all": "none"}.get(dom, "worker_min_max")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = os.environ.get("SMOKE") == "1"
    coh = X.build_cohort()
    if smoke:
        coh = coh.sample(200, random_state=0).reset_index(drop=True)      # ~200 images, all domains
    else:
        assert len(coh) == 5889, len(coh)
    N = len(coh)
    ann, det = E.prepare(coh)                                              # fills E.G (KP, CF, POSE, R, VALID, WK, CTX, DOM, PID)
    dom_arr, wk = np.array(G["DOM"]), G["WK"]
    metrics = {"n_images": N, "smoke": int(smoke), "n_boot": NB, "S_max": S_MAX, "S_angle": S_ANG, "mc_image_subsample": 0,
               "sigma_fraction_pipeline": DEFAULT_SIGMA_FRACTION}
    for d in DOMAINS[1:]:
        metrics[f"n_images/{d}"], metrics[f"n_workers/{d}"] = int((dom_arr == d).sum()), len(set(wk[dom_arr == d]))
    rows_of = {d: np.arange(N) if d == "all" else np.flatnonzero(dom_arr == d) for d in DOMAINS}

    # ---- annotated poses, pipeline ref_length, normalised errors, angle differences (annotation vs prediction)
    lps = [EA.label_pose(r) for r in coh.to_dict("records")]
    ac = np.array([lp.confidence for lp in lps])
    ref = np.array([compute_anthropometry(p).ref_length for p in G["POSE"]])
    rp = np.linalg.norm(G["KP"][:, NECK, :2] - G["KP"][:, PELVIS, :2], axis=1)
    use = G["VALID"] & (rp > 0)
    err = normalized_error(np.where(use[:, None, None], G["R"], 0.0), np.where(use, rp, 1.0), ref, (ac[:, J] > 0) & use[:, None])
    cf = G["CF"][:, J]
    metrics["pj/n_images_with_residual"] = int(use.sum())
    metrics["pj/definition_pelvis_relative"] = 1          # errors are measured after aligning pelvis origin and torso scale (E.residual)
    adiff = np.full((N, len(ANG)), np.nan)
    for i in range(N):
        need = np.array([all(ac[i, j] > 0 for j in nj) for nj in ANG_NEED])
        adiff[i] = np.where(need, np.abs((angles_of(G["POSE"][i]) - angles_of(lps[i]) + 180.0) % 360.0 - 180.0), np.nan)  # 검수 E1: 각도 차이는 -180~180 으로 접은 뒤 절댓값(359도 어긋남은 1도 차이)
    print(f"PWW_PROGRESS 0/5 prepared N={N}", flush=True)

    # ---- items 1-4
    pj_rows = []
    names = pj_names()
    for dom in DOMAINS:
        pt, lo, hi = summarize(dom, wk, rows_of[dom], lambda r: pj_stat(err, cf, r))
        record(pj_rows, metrics, "pj", dom, [(s, j) for s, j in names], pt, lo, hi, interval_label(dom))
    pd.DataFrame(pj_rows).rename(columns={"k0": "stat", "k1": "joint"}).to_csv(OUT / "per_joint.csv", index=False)
    print("PWW_PROGRESS 1/5 per-joint error", flush=True)

    ctx = get_context("fork")
    with ctx.Pool(12) as pool:
        # ---- item 5: sd (degrees) of each angle under each perturbation size
        asd = {s: np.full((N, len(ANG)), np.nan) for s in SIGMAS}
        asd_xy = {s: np.full((N, len(ANG)), np.nan) for s in SIGMAS}
        for s in SIGMAS:
            for i, v, vz in pool.imap_unordered(_w_angle_sd, [(i, s) for i in range(N)], chunksize=20):
                asd[s][i], asd_xy[s][i] = v, vz
        print("PWW_PROGRESS 2/5 angle sd", flush=True)
        # ---- items 6-7: every perturbed pose scored once per (sigma, seed 0) and (0.04, seed 1)
        L = {(s, 0): np.full((N, S_MAX, len(OUTS)), -1, np.int8) for s in SIGMAS}
        L[(DEFAULT_SIGMA_FRACTION, 1)] = np.full((N, S_MAX, len(OUTS)), -1, np.int8)
        LZ = {s: np.full((N, S_ANG, len(OUTS)), -1, np.int8) for s in SIGMAS}     # item 8: z restored, S = 128
        tasks = [(i, s, sd, False) for (s, sd) in L for i in range(N)] + [(i, s, 0, True) for s in SIGMAS for i in range(N)]
        for n, ((i, s, sd, zf), lev) in enumerate(pool.imap_unordered(_w_levels, tasks, chunksize=8)):
            (LZ[s] if zf else L[(s, sd)])[i] = lev
            if n % 2000 == 0:
                print(f"PWW_PROGRESS {n}/{len(tasks)} perturbed scoring", flush=True)
    print("PWW_PROGRESS 3/5 scoring done", flush=True)

    # ---- item 5 tables
    cols, kinds, keys = [], [], []
    for s in SIGMAS:
        for a in range(len(ANG)):
            for kd in ("median", "rms", "p95"):
                cols.append(asd[s][:, a]), kinds.append(kd), keys.append(("pert", ANG[a], f"sigma{sg(s)}", kd))
    for a in range(len(ANG)):
        for kd in ("n", "median", "mean", "p68", "p95"):
            cols.append(adiff[:, a]), kinds.append(kd), keys.append(("diff", ANG[a], kd))
    Q = np.column_stack(cols)
    ang_rows = []
    for dom in DOMAINS:
        pt, lo, hi = summarize(dom, wk, rows_of[dom], lambda r: col_stats(Q[r], kinds))
        record(ang_rows, metrics, "angle", dom, keys, pt, lo, hi, interval_label(dom))
    pd.DataFrame(ang_rows).rename(columns={"k0": "kind", "k1": "angle", "k2": "sigma_or_stat", "k3": "stat"}).to_csv(OUT / "angle_deg.csv", index=False)

    # ---- item 6: Monte Carlo standard error and seed-to-seed agreement at the pipeline sigma, S in SAMPLE_COUNTS
    cols, kinds, keys = [], [], []
    for name, f in OUTS:
        lo_c, n_c = E.FIELDS[f][1], E.FIELDS[f][2]
        for S in SAMPLE_COUNTS:
            (P0, t0), (P1, _) = [class_probs(L[(DEFAULT_SIGMA_FRACTION, sd)][:, :, f], S, lo_c, n_c) for sd in SEEDS_PAIR]
            se, dis, dp = mc_se(P0.max(1), t0), (P0.argmax(1) != P1.argmax(1)).astype(float), np.abs(P0.max(1) - P1.max(1))
            for vec, kd, q in ((se, "mean", "se_mean"), (se, "p95", "se_p95"), (dis, "mean", "disagree"), (dp, "mean", "dp_mean"), (dp, "p95", "dp_p95")):
                cols.append(vec), kinds.append(kd), keys.append((name, f"S{S}", q))
    Q = np.column_stack(cols)
    cnt_rows = []
    for dom in DOMAINS:
        pt, lo, hi = summarize(dom, wk, rows_of[dom], lambda r: col_stats(Q[r], kinds))
        record(cnt_rows, metrics, "count", dom, keys, pt, lo, hi, interval_label(dom))
    pd.DataFrame(cnt_rows).rename(columns={"k0": "output", "k1": "S", "k2": "stat"}).to_csv(OUT / "sample_count.csv", index=False)
    print("PWW_PROGRESS 4/5 sample count", flush=True)

    # ---- item 7: action-level retention and accuracy against the annotation-derived action level
    cols, kinds, keys = [], [], []
    for m, f in ACT:
        lo_c, n_c = E.FIELDS[f][1], E.FIELDS[f][2]
        dl, al = det[:, f], ann[:, f]
        cols.append((dl == al).astype(float)), kinds.append("mean"), keys.append((m, "deterministic", "acc_det"))
        for s in SIGMAS:
            for S in SAMPLE_COUNTS:
                P, _ = class_probs(L[(s, 0)][:, :, f], S, lo_c, n_c)
                mode = P.argmax(1) + lo_c
                share = P[np.arange(N), np.clip(dl - lo_c, 0, n_c - 1)]
                for vec, q in (((mode == dl).astype(float), "keep_mode"), (share, "keep_share"), ((mode == al).astype(float), "acc_mode")):
                    cols.append(vec), kinds.append("mean"), keys.append((m, f"sigma{sg(s)}", f"S{S}", q))
    Q = np.column_stack(cols)
    sens_rows = []
    for dom in DOMAINS:
        pt, lo, hi = summarize(dom, wk, rows_of[dom], lambda r: col_stats(Q[r], kinds))
        record(sens_rows, metrics, "sens", dom, keys, pt, lo, hi, interval_label(dom))
    sd_ = pd.DataFrame(sens_rows)
    sd_["method"], sd_["sigma"] = sd_.k0, sd_.k1
    sd_["S"] = sd_.k2.where(sd_.k2.str.startswith("S"), "")
    sd_["quantity"] = sd_.k2.where(~sd_.k2.str.startswith("S"), sd_.k3)
    sd_[["domain", "method", "sigma", "S", "quantity", "point", "lo", "hi", "interval"]].to_csv(OUT / "output_sensitivity.csv", index=False)

    # ---- item 8: does the z-column perturbation matter? xyz (pipeline, the reported result) next to xy only
    cols, kinds, keys = [], [], []
    zr = np.array([z_range_ratio(G["KP"][i]) for i in range(N)])
    for kd in ("median", "p5", "p95"):
        cols.append(zr), kinds.append(kd), keys.append(("z_range_over_xy_range", kd))
    for s in SIGMAS:
        for a in range(len(ANG)):
            for kd in ("median", "rms", "p95"):
                for var, src in (("xyz", asd), ("xy", asd_xy)):
                    cols.append(src[s][:, a]), kinds.append(kd), keys.append(("angle_sd", var, ANG[a], f"sigma{sg(s)}", kd))
    S8 = SAMPLE_COUNTS.index(128)
    for m, f in ACT:
        lo_c, n_c = E.FIELDS[f][1], E.FIELDS[f][2]
        dl, al = det[:, f], ann[:, f]
        for s in SIGMAS:
            q = {}
            for var, lev in (("xyz", L[(s, 0)][:, :, f]), ("xy", LZ[s][:, :, f])):
                P, _ = class_probs(lev, 128, lo_c, n_c)
                mode = P.argmax(1) + lo_c
                q[var] = {"keep_mode": (mode == dl).astype(float), "keep_share": P[np.arange(N), np.clip(dl - lo_c, 0, n_c - 1)],
                          "acc_mode": (mode == al).astype(float)}
                for qn, vec in q[var].items():
                    cols.append(vec), kinds.append("mean"), keys.append(("level", var, m, f"sigma{sg(s)}", qn))
            for qn in ("keep_mode", "keep_share"):
                cols.append(q["xyz"][qn] - q["xy"][qn]), kinds.append("mean"), keys.append(("level_diff_xyz_minus_xy", m, f"sigma{sg(s)}", qn))
    Q = np.column_stack(cols)
    z_rows = []
    for dom in DOMAINS:
        pt, lo, hi = summarize(dom, wk, rows_of[dom], lambda r: col_stats(Q[r], kinds))
        record(z_rows, metrics, "zfx", dom, keys, pt, lo, hi, interval_label(dom))
    pd.DataFrame(z_rows).to_csv(OUT / "z_effect.csv", index=False)
    metrics["zfx/mc_image_subsample"] = 0

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("PWW_PROGRESS 5/5 done", flush=True)
    for d in DOMAINS:
        print(d, "ratio_median", round(metrics.get(f"pj/{d}/ratio_median/all/point", float("nan")), 3))


if __name__ == "__main__":
    main()
