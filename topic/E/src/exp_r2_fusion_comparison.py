# PWW-DISCLOSURE: public
"""exp-r2-fusion-comparison (v2, audit T2/T4/E2 fix): single backbone vs soft / max / guarded fusion (reviewer 5-3).

Pre-registered design: topic/E/paper/targets/targets.md, section "exp-r2-fusion-comparison".
Frozen set: test_split_20.csv (1,178 images, sha256 checked). Reference: the dataset's
annotated keypoints of the labelled worker, and the OWAS/RULA/REBA action levels computed
from them with the same scorer (eval_action_agreement.levels_of).

Two stages, because the GPU env (onnxruntime-gpu) has no scikit-learn and the metric env has
no GPU:
  detect    (env bioera_E_posecmp, GPU)  run the three backbones once per image and cache ALL
            their raw detections in $PWW_RESULT_DIR/detections.npz
  evaluate  (env bioera_mirror_test)     replay the cache through the unchanged fusion code in
            ace_pose/bioera_v2.py, so every mode is computed from the same detections.
Fusion code is reused, not re-implemented: `ensemble_inference` (soft, max; each backbone
picks its own primary person, as the manuscript describes) and `infer_from_v1_baseline`
(guarded; RTMW3D labelled-worker pose is the spine). The "single" method is the RTMW3D
detection that matches the labelled worker.

v2 evaluates two conditions (targets.md "exp-r2-fusion-comparison-v2"): same_person (every method on the
labelled worker's detection of each backbone) and no_annotation (no method sees the annotation).
The old person-selection note above describes v1; fuse_all() documents v2.

Usage: python exp_r2_fusion_comparison.py {detect|evaluate} [--limit N]
Writes (to $PWW_RESULT_DIR): detections.npz, metrics.json, summary_table.csv, per_image.csv.
Exploratory additions (evaluate; explor_boot/strata/cache_mismatch.csv, metrics 'explor/...'): worker-level bootstrap of the
PCK / action-agreement differences, single- vs multi-person strata, soft weight exp(k*confidence) for k = 1, 3, 10 (computed here by
soft_k_fuse because ace_pose hard-codes k = 3), classification of the cohort-cache mismatch. The image-level bootstrap above is kept.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "topic/E/result/exp-r2-fusion-comparison-v2"))
# v1 cache holds ALL persons of every backbone; v2 reads it read-only instead of re-detecting
# 코호트 자세 캐시와 정답 관절점 표는 데이터 등록부를 거쳐 읽는다(검수 D1)
from pww_data import resolve  # noqa: E402
COHORT = Path(resolve("cohort_pose_cache"))
DET_NPZ = Path(resolve("fusion_detections"))
SPLIT = Path(resolve("action_agreement_test_split"))
SPLIT_SHA = SPLIT.with_name("test_split_20.sha256")
METHODS = ["single", "soft", "max", "guarded"]
STDS = ["OWAS", "RULA", "REBA"]
LEVEL_SETS = {"OWAS": [1, 2, 3, 4], "RULA": [1, 2, 3, 4], "REBA": [0, 1, 2, 3, 4]}
N_BOOT, SEED = 2000, 0
SOFT_KS = (1, 3, 10)                     # exploratory: soft weight exp(k * confidence); the manuscript's soft fusion is k = 3
XM = [f"soft_k{k}" for k in SOFT_KS]     # extra methods, evaluated beside METHODS but kept out of the pre-registered tables
ALLM = METHODS + XM


# ---------------------------------------------------------------- pure helpers
def qwk(ref, sys_, levels):
    """Quadratic-weighted kappa from level arrays (same as sklearn weights='quadratic')."""
    k = len(levels)
    r, s = np.searchsorted(levels, ref), np.searchsorted(levels, sys_)
    o = np.bincount(r * k + s, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / max(k - 1, 1) ** 2
    return float(1 - (w * o).sum() / (w * e).sum()) if (w * e).sum() > 0 else float("nan")


def paired_ci(stat, idx):
    """stat(index_array) -> float. Returns (point, lo, hi) of stat over the bootstrap resamples
    idx (B x n, shared by every comparison so all differences are paired)."""
    n = idx.shape[1]
    point = stat(np.arange(n))
    b = np.array([stat(i) for i in idx])
    return float(point), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def check_hash():
    got = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    want = SPLIT_SHA.read_text().split()[0]
    if got != want:
        sys.exit(f"frozen split hash mismatch: {got} != {want}")


# ---------------------------------------------------------------- stage 1: detect
def detect(limit):
    import cv2
    import pandas as pd
    import ace_pose.bioera_v2 as v2
    from ace_pose.data_paths import IMG_DIR

    check_hash()
    stems = pd.read_csv(SPLIT).file_stem.tolist()[:limit]
    files = {p.stem: p for p in Path(IMG_DIR).iterdir()}
    models = {"wb": v2._wb3d(), "b": v2._body(), "bf": v2._bodyfeet()}
    cache, fail, t0 = {}, 0, time.time()
    for i, st in enumerate(stems):
        img = cv2.imread(str(files[st]))
        w = v2._safe_call(models["wb"], img)          # (k3, scores, simcc, kpts2d)
        b = v2._safe_call(models["b"], img)           # (kpts, scores)
        f = v2._safe_call(models["bf"], img)
        # raw detections of every person; absent output -> empty arrays (as the fusion code expects)
        cache[f"{st}/wb_k"] = w[3][:, :23].astype(np.float32) if w is not None else np.zeros((0, 23, 2), np.float32)
        cache[f"{st}/wb_s"] = w[1][:, :23].astype(np.float32) if w is not None else np.zeros((0, 23), np.float32)
        for tag, o, nk in (("b", b, 17), ("bf", f, 23)):
            cache[f"{st}/{tag}_k"] = o[0].astype(np.float32) if o else np.zeros((0, nk, 2), np.float32)
            cache[f"{st}/{tag}_s"] = o[1].astype(np.float32) if o else np.zeros((0, nk), np.float32)
        fail += w is None
        if (i + 1) % 100 == 0:
            print(f"PWW_PROGRESS {i + 1}/{len(stems)}  {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "detections.npz", stems=np.array(stems), **cache)
    print(f"detect done: {len(stems)} images, rtmw3d failures {fail}, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- stage 2: evaluate
class Replay:
    """Stands in for a rtmlib model: returns the cached raw output of the current image."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, _img):
        return self.fn()


SAME, NOANN = "same_person", "no_annotation"
CONDS = [SAME, NOANN]


def soft_k_fuse(arr, k, v2):
    """Soft fusion of the 23 joints with weight exp(k * confidence), computed here because ace_pose.bioera_v2.ensemble_inference
    hard-codes k = 3 (that file is read-only). Same candidates and primary-person choice as ensemble_inference(fusion_mode='soft');
    the unit test pins k = 3 to its output. arr = {'wb'|'b'|'bf': (keypoints, scores)}. None when RTMW3D found nobody."""
    (wk, ws), (bk, bs), (fk, fs) = arr["wb"], arr["b"], arr["bf"]
    if len(wk) == 0:
        return None
    p = v2._primary_idx(ws, kpts=wk, target_centroid=None)
    k_wb, sc_wb = wk[p], ws[p]
    k_b, sc_b = np.zeros((17, 2)), np.zeros(17)
    if len(bk):
        p = v2._primary_idx(bs, sub_idx=slice(0, 17), kpts=bk, target_centroid=None)
        k_b, sc_b = bk[p], bs[p]
    k_bf, sc_bf = np.zeros((23, 2)), np.zeros(23)
    if len(fk):
        p = v2._primary_idx(fs, sub_idx=slice(0, 17), kpts=fk, target_centroid=None)
        k_bf, sc_bf = fk[p], fs[p]
    pos, sco = np.zeros((23, 2)), np.zeros(23)
    for j in range(23):
        c = [(sc_wb[j], k_wb[j])] if sc_wb[j] > 0 else []
        if j < 17 and sc_b[j] > 0:
            c.append((sc_b[j], k_b[j]))
        if sc_bf[j] > 0:
            c.append((sc_bf[j], k_bf[j]))
        if not c:
            pos[j] = k_wb[j]
            continue
        conf = np.array([x for x, _ in c])
        w = np.exp(conf * float(k))
        w = w / w.sum()
        pos[j], sco[j] = sum(w[i] * c[i][1] for i in range(len(c))), conf.max()
    return pos, sco


def fuse_all(det, stem, ann, cond, v2, label_distance, ks=()):
    """Four pose variants (23x2 positions, 23 scores, log) for one image from the cached detections.

    SAME  (condition 가): for EVERY backbone keep only the detection nearest to the labelled worker,
          then run all four methods on those single-person detections.
    NOANN (condition 나): `ann` is never read. single = RTMW3D primary detection, guarded is locked to
          it, soft / max let each backbone take its own primary person.
    Returns (out, n_det, sel); sel = chosen person index per backbone (SAME) or None (NOANN).
    """
    arr = {t: (det[f"{stem}/{t}_k"], det[f"{stem}/{t}_s"]) for t in ("wb", "b", "bf")}
    n_det, sel = len(arr["wb"][0]), None
    if cond == SAME:
        sel = {t: int(np.argmin([label_distance(q[:17], ann) for q in k])) if len(k) else None
               for t, (k, _) in arr.items()}
        arr = {t: (k[sel[t]:sel[t] + 1], s[sel[t]:sel[t] + 1]) if sel[t] is not None else (k, s)
               for t, (k, s) in arr.items()}
    (wk, ws), (bk, bs), (fk, fs) = arr["wb"], arr["b"], arr["bf"]
    v2._Models.wb3d = Replay(lambda: (np.zeros((len(wk), 1, 3)), ws, None, wk))
    v2._Models.body = Replay(lambda: (bk, bs))
    v2._Models.bodyfeet = Replay(lambda: (fk, fs))
    # single: RTMW3D person (the labelled one under SAME, the primary one under NOANN)
    p = v2._primary_idx(ws, kpts=wk)
    k25 = np.zeros((25, 2)); s25 = np.zeros(25)
    k25[:23], s25[:23] = wk[p], ws[p]
    k25[23], s25[23] = 0.5 * (wk[p, 5] + wk[p, 6]), min(ws[p, 5], ws[p, 6])
    k25[24], s25[24] = 0.5 * (wk[p, 11] + wk[p, 12]), min(ws[p, 11], ws[p, 12])
    out = {"single": (k25[:23], s25[:23], None)}
    for m in ("soft", "max"):
        r = v2.ensemble_inference(None, fusion_mode=m)
        out[m] = (r["kpts"][:23], r["scores"][:23], None) if r else None
    r = v2.infer_from_v1_baseline(None, k25, s25)
    out["guarded"] = (r["kpts"][:23], r["scores"][:23], r["verification_log"][:23])
    for k in ks:                                     # exploratory soft-weight sensitivity
        r = soft_k_fuse(arr, k, v2)
        out[f"soft_k{k}"] = (r[0], r[1], None) if r else None
    return out, n_det, sel


def _same_out(a, b):
    return all((a[m] is None) == (b[m] is None) and (a[m] is None or np.array_equal(a[m][0], b[m][0]))
               for m in METHODS)


def invariance_check(det, stems, ann_of_stem, v2, label_distance, n=40):
    """Run-time audit assertions (code audit T2/T4/E2) on the first n multi-person images.
    NOANN: moving the annotation anywhere must not change any method's keypoints.
    SAME : changing detections of the NON-selected persons must not change any method's keypoints
           (all four methods use the same labelled person per backbone)."""
    rng, done = np.random.default_rng(1), 0
    for st in stems:
        if len(det[f"{st}/wb_k"]) < 2:
            continue
        ann = ann_of_stem(st)
        far = {**ann, "keypoints": ann["keypoints"] + np.array([700.0, -300.0, 0.0])}
        a = fuse_all(det, st, ann, NOANN, v2, label_distance)[0]
        b = fuse_all(det, st, far, NOANN, v2, label_distance)[0]
        assert _same_out(a, b), f"{st}: NOANN output depends on annotation"
        base, sel = fuse_all(det, st, ann, SAME, v2, label_distance)[::2]
        alt = dict(det)
        for t in ("wb", "b", "bf"):
            k = det[f"{st}/{t}_k"].copy()
            for q in range(len(k)):
                if q != sel[t]:
                    k[q] += rng.normal(0, 50, k[q].shape).astype(np.float32)
            alt[f"{st}/{t}_k"] = k
        assert _same_out(base, fuse_all(alt, st, ann, SAME, v2, label_distance)[0]), \
            f"{st}: SAME output depends on a non-selected person"
        done += 1
        if done == n:
            break
    return done


TOL_SAME, TOL_NUM = 1e-3, 2.0     # px: "identical" below TOL_SAME; a cached pose within TOL_NUM of a detection is that detection


def cache_mismatch_row(stem, wb_k, cached, sel, diff):
    """Classify why the selected RTMW3D detection differs from the cohort-cache pose (diff = largest joint offset in px).
    identical: diff < TOL_SAME. person_selection: the cache pose matches ANOTHER detection of the image. numeric_same_person: it is
    nearest to the selected detection (different numbers for the same person). unmatched: near no detection."""
    d = np.abs(wb_k - cached[None]).max(axis=(1, 2))        # per detection
    q = int(np.argmin(d))
    cat = ("identical" if diff < TOL_SAME else "numeric_same_person" if q == sel else
           "person_selection" if d[q] <= TOL_NUM else "unmatched")
    return dict(file_stem=stem, n_det=len(wb_k), selected=sel, nearest=q, diff_selected_px=diff, diff_nearest_px=float(d[q]), category=cat)


PAIRS = [("soft", "single"), ("max", "single"), ("guarded", "single"), ("guarded", "soft"), ("guarded", "max"),
         *[(x, "single") for x in XM], *[("guarded", x) for x in XM]]
PCK_ACC = ["pck"] + [f"{s}_acc" for s in STDS]


def explor_fusion(E, S, wkey, cnt, ref, ndet, mismatch, res):
    """Exploratory additions (reviewer-driven, no pass/fail): (1) worker-level bootstrap of the PCK and action-level-agreement
    differences (the image-level paired bootstrap above is kept), including the soft-weight variants; (2) single- vs multi-person
    scene strata with the identity-swap rate; (3) classification of the cohort-cache mismatch. Writes explor_*.csv, fills `res`."""
    import pandas as pd
    per = {c: {m: np.array([np.array(S[c]["lv"][m][s]) == ref[s] for s in STDS], float) for m in ALLM} for c in CONDS}  # (3, n) level hits
    hits = {c: {m: np.array(S[c]["hits"][m], float) for m in ALLM} for c in CONDS}

    def vals(c, m, ix):     # pooled PCK (as in the main table: hits / valid joints), then level agreement
        return np.array([hits[c][m][ix].sum() / cnt[ix].sum(), *per[c][m][:, ix].mean(1)])

    def stat(ix):
        v = {(c, m): vals(c, m, ix) for c in CONDS for m in ALLM}
        return np.concatenate([v[(c, a)] - v[(c, b)] for c in CONDS for a, b in PAIRS])
    point, lo, hi = E.wk_boot(wkey, stat)
    names = [(c, f"{a}_minus_{b}", nm) for c in CONDS for a, b in PAIRS for nm in PCK_ACC]
    boot = [dict(analysis="fusion_worker_boot", condition=c, pair=pr, stat=nm, point=point[j], ci_lo=lo[j], ci_hi=hi[j],
                 n_images=len(wkey), n_workers=len(set(wkey)), n_boot=E.N_BOOT_WK) for j, (c, pr, nm) in enumerate(names)]
    pd.DataFrame(boot).to_csv(OUT / "explor_boot.csv", index=False)
    E.put_metrics(res, boot, ("analysis", "condition", "pair", "stat"))
    strata = []
    for c in CONDS:
        for m in ALLM:
            sw = np.array(S[c]["swap"][m])
            for name, mask in (("single_person", ndet <= 1), ("multi_person", ndet >= 2)):
                ix = np.flatnonzero(mask)
                ok = sw[ix] >= 0
                strata.append(dict(condition=c, method=m, stratum=name, n_images=len(ix), **dict(zip(PCK_ACC, vals(c, m, ix))),
                                   swap_n=int(sw[ix][ok].sum()), swap_rate=float(sw[ix][ok].mean()) if ok.any() else float("nan")))
    pd.DataFrame(strata).to_csv(OUT / "explor_strata.csv", index=False)
    for r in strata:
        for k in (*PCK_ACC, "swap_rate", "n_images"):
            res[f"explor/strata/{r['condition']}/{r['method']}/{r['stratum']}/{k}"] = r[k]
    mm = pd.DataFrame(mismatch)
    mm.to_csv(OUT / "explor_cache_mismatch.csv", index=False)
    for cat, g in mm.groupby("category"):
        res[f"explor/cache_mismatch/{cat}/n"] = len(g)
        res[f"explor/cache_mismatch/{cat}/n_multi_person"] = int((g.n_det >= 2).sum())
        res[f"explor/cache_mismatch/{cat}/median_diff_px"] = float(g.diff_selected_px.median())
    res["explor/cache_mismatch/n_images"] = len(mm)


def evaluate(limit):
    import pandas as pd
    import ace_pose.bioera_v2 as v2
    from ace_pose.batch_infer import LABEL_MATCH_MAX, label_distance
    from compare_pose_models_eval import JOINTS, ann_of, m25_from_2d
    from eval_action_agreement import label_pose, levels_of

    import exp_r2_empirical_noise as E    # lazy: pulls the cohort/ace_pose stack, which the GPU detect env lacks; wk_boot is shared
    check_hash()
    det = np.load(DET_NPZ)
    stems = [str(s) for s in det["stems"]][:limit]
    summ = pd.read_csv(COHORT / "summary.csv").set_index("file_stem")
    lab = pd.read_csv(resolve("aihub_busan_annotations"))
    lab["file_stem"] = lab.file_name.str[:-4]
    lab = lab.drop_duplicates(["file_stem", "worker"], keep=False).set_index(["file_stem", "worker"])

    def row_of(st):
        return lab.loc[(st, int(summ.at[st, "worker_id"]))]

    n_inv = invariance_check(det, stems, lambda st: ann_of(row_of(st)), v2, label_distance)
    print(f"invariance assertions passed on {n_inv} multi-person images", flush=True)

    rows, cnt, cache_diff, mismatch, ndet = [], [], [], [], []
    lv_ref = {s: [] for s in STDS}
    S = {c: dict(hits={m: [] for m in ALLM}, lv={m: {s: [] for s in STDS} for m in ALLM},
                 changed={m: [0, 0] for m in ALLM}, improved=[0, 0], subst={m: [0, 0] for m in ALLM}, swap={m: [] for m in ALLM})
         for c in CONDS}
    wkey = np.array([f"{summ.at[st, 'factory_id']}_{summ.at[st, 'worker_id']}" for st in stems])
    for st in stems:
        row = row_of(st)
        ann = ann_of(row)
        diag = max(np.hypot(row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1), 1.0)
        outs = {}
        for c in CONDS:
            outs[c], n_det, sel_c = fuse_all(det, st, ann, c, v2, label_distance, SOFT_KS)
            if c == SAME:
                sel = sel_c        # labelled-person index per backbone (NOANN has none)
        z = np.load(COHORT / "details" / f"{st}.npz")   # read-only consistency check
        cache_diff.append(float(np.abs(outs[SAME]["single"][0] - z["kpts2d_133"][:23]).max()))
        mismatch.append(cache_mismatch_row(st, det[f"{st}/wb_k"], z["kpts2d_133"][:23], sel["wb"], cache_diff[-1]))
        ndet.append(n_det)
        ref_lv = levels_of(label_pose(row.to_dict()), summ.at[st, "process_label"])
        for s in STDS:
            lv_ref[s].append(ref_lv[s])
        valid = [(cj, l) for cj, l in JOINTS.items() if ann["keypoints"][l, 2] > 0]
        cnt.append(len(valid))
        rec = {"file_stem": st, "n_det": n_det, "explor_n_valid_joints": len(valid)}
        for c in CONDS:
            out, A = outs[c], S[c]
            for m in ALLM:
                if out[m] is None:                       # fusion produced no skeleton: every joint missed
                    A["hits"][m].append(0)
                    for s in STDS:
                        A["lv"][m][s].append(-99)
                        rec[f"explor_{c}/{m}_{s}_ok"] = 0
                    A["swap"][m].append(-1)              # no skeleton: identity swap undefined
                    rec[f"{c}/{m}_pck_hits"], rec[f"explor_{c}/{m}_swapped"] = 0, -1
                    continue
                k, s_, _ = out[m]
                err = [np.hypot(*(k[cj] - ann["keypoints"][l, :2])) / diag for cj, l in valid]
                A["hits"][m].append(int(sum(e <= 0.05 for e in err)))
                lvm = levels_of(m25_from_2d(k, s_), summ.at[st, "process_label"])
                for s in STDS:
                    A["lv"][m][s].append(lvm[s])
                    rec[f"explor_{c}/{m}_{s}_ok"] = int(lvm[s] == ref_lv[s])
                swapped = int(label_distance(k[:17], ann) > LABEL_MATCH_MAX)   # skeleton not on the labelled worker (any scene)
                A["swap"][m].append(swapped)
                rec[f"explor_{c}/{m}_swapped"] = swapped
                if n_det >= 2:                           # identity substitution (pre-registered count: multi-person scenes only)
                    A["subst"][m][1] += 1
                    A["subst"][m][0] += swapped
                mv = np.hypot(*(k - out["single"][0]).T) > 1e-3
                A["changed"][m][0] += int(mv.sum()); A["changed"][m][1] += 23
                rec[f"{c}/{m}_pck_hits"] = A["hits"][m][-1]
            if out["guarded"][2] is not None:
                A["improved"][0] += sum(x == "v2-improved" for x in out["guarded"][2]); A["improved"][1] += 23
        rows.append(rec)

    n = len(stems)
    cnt = np.array(cnt)
    ref = {s: np.array(v) for s, v in lv_ref.items()}
    idx = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))   # shared resamples -> paired, same in both conditions
    names = ["pck"] + [f"{s}_{k}" for s in STDS for k in ("acc", "qwk")]
    res = {"n_images": n, "split_sha256_ok": 1, "bootstrap_resamples": N_BOOT, "seed": SEED,
           "invariance_images_checked": n_inv, "invariance_ok": 1,
           "detections_source": str(DET_NPZ.relative_to(ROOT)),
           "detections_sha256": hashlib.sha256(DET_NPZ.read_bytes()).hexdigest()}
    table = []
    for c in CONDS:
        A = S[c]
        hits = {m: np.array(h) for m, h in A["hits"].items()}
        lv = {m: {s: np.array(v) for s, v in d.items()} for m, d in A["lv"].items()}

        def stat(m, name):
            if name == "pck":
                return lambda i: hits[m][i].sum() / cnt[i].sum()
            std = name.split("_")[0]
            if name.endswith("acc"):
                return lambda i: float((lv[m][std][i] == ref[std][i]).mean())
            return lambda i: qwk(ref[std][i], np.clip(lv[m][std][i], LEVEL_SETS[std][0], LEVEL_SETS[std][-1]), LEVEL_SETS[std])

        for m in METHODS:
            row = {"condition": c, "method": m, "n": n}
            for nm in names:
                v = stat(m, nm)(np.arange(n)); res[f"{c}/{m}/{nm}"] = v; row[nm] = v
            sub = A["subst"][m]
            row["identity_subst_rate"] = sub[0] / sub[1] if sub[1] else float("nan")
            row["identity_subst_n"], row["multi_person_images"] = sub
            row["joints_changed_vs_single"] = A["changed"][m][0] / A["changed"][m][1] if A["changed"][m][1] else float("nan")
            res[f"{c}/{m}/subst_rate"], res[f"{c}/{m}/subst_n"], res[f"{c}/{m}/multi_person_images"] = row["identity_subst_rate"], sub[0], sub[1]
            res[f"{c}/{m}/joints_changed"] = row["joints_changed_vs_single"]
            table.append(row)
        res[f"{c}/guarded/joints_v2_improved"] = A["improved"][0] / A["improved"][1]
        for a, b in (("soft", "single"), ("max", "single"), ("guarded", "single"), ("guarded", "soft"), ("guarded", "max")):
            row = {"condition": c, "method": f"{a}_minus_{b}", "n": n}
            for nm in names:
                fa, fb = stat(a, nm), stat(b, nm)
                d, lo, hi = paired_ci(lambda i: fa(i) - fb(i), idx)
                k0 = f"{c}/{a}_minus_{b}/{nm}"
                res[k0], res[k0 + "_ci_lo"], res[k0 + "_ci_hi"] = d, lo, hi
                row[nm], row[f"{nm}_ci_lo"], row[f"{nm}_ci_hi"] = d, lo, hi
            table.append(row)

    # pre-registered criterion (a)(b), unchanged, applied mechanically to condition (가); (나) is reported only
    g = f"{SAME}/guarded_minus_single"
    a_pck = res[f"{g}/pck_ci_lo"] > -0.01
    a_lvl = {s: res[f"{g}/{s}_acc_ci_lo"] > -0.02 for s in STDS}
    b_soft = res[f"{SAME}/guarded_minus_soft/pck_ci_lo"] > 0
    b_max = res[f"{SAME}/guarded_minus_max/pck_ci_lo"] > 0
    res.update({"criterion_a_pck": int(a_pck), **{f"criterion_a_{s}": int(v) for s, v in a_lvl.items()},
                "criterion_a": int(a_pck and all(a_lvl.values())),
                "criterion_b_vs_soft": int(b_soft), "criterion_b_vs_max": int(b_max), "criterion_b": int(b_soft and b_max)})
    res["single_vs_batch_v1_cache_max_abs_px_median"] = float(np.median(cache_diff))
    res["single_vs_batch_v1_cache_max_abs_px_max"] = float(np.max(cache_diff))
    res["single_vs_batch_v1_cache_identical_share"] = float(np.mean(np.array(cache_diff) < 1e-3))
    OUT.mkdir(parents=True, exist_ok=True)
    explor_fusion(E, S, wkey, cnt, ref, np.array(ndet), mismatch, res)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    df = pd.DataFrame(table)
    df.to_csv(OUT / "summary_table.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "per_image.csv", index=False)
    print(df[["condition", "method", "pck", "OWAS_acc", "RULA_acc", "REBA_acc", "identity_subst_rate"]].to_string())
    print({k: v for k, v in res.items() if k.startswith("criterion")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["detect", "evaluate"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"detect": detect, "evaluate": evaluate}[a.stage](a.limit)
