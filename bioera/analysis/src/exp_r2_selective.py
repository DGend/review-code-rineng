# PWW-DISCLOSURE: public
"""exp-r2-selective: selective accuracy on held-out workers (targets.md, "exp-r2-selective").

Why: if only confident predictions are accepted, is the accepted share accurate against the annotation-derived
score? Per fold the threshold t is the smallest training-image confidence whose accepted training accuracy is
>= 0.96; it is applied unchanged to the held-out workers. Confidence = top probability of the A1 (empirical
residual) distribution, or of the current Gaussian-jitter distribution. Training images use donors that exclude
their own worker (calibration mode of exp-r2-empirical-noise); evaluation images use the fold's training donors.
The decision aggregate is the MEAN OVER FOLDS. Everything is imported from exp_r2_empirical_noise (no copy).

Exploratory additions after the internal review (metrics keys 'explor/...', files explor_*.csv, no pass/fail): accuracy at fixed
acceptance shares, AURC against the Gaussian and random order, acceptance coverage, error structure, the training-accuracy target
sweep (0.93-0.97, both aggregation rules) and a worker bootstrap of the accepted accuracy and share. SMOKE=<n> subsamples n images.
"""
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_r2_empirical_noise as E  # noqa: E402

X, EA, G = E.X, E.EA, E.G
OUT = Path(os.environ.get("PWW_RESULT_DIR", E.ROOT / "bioera/analysis/result/exp-r2-selective"))
MIN_ACC, PASS_ACC, PASS_CELLS, NARROW_SHARE = 0.96, 0.93, 7, 0.20
# exploratory (reviewer-driven, post hoc, no pass/fail): fixed acceptance shares, training-accuracy targets
SHARES, MIN_ACC_SWEEP, GRID = (0.1, 0.2, 0.3, 0.5, 0.8), (0.93, 0.95, 0.96, 0.97), np.linspace(0.01, 1, 100)


def select_threshold(conf, correct, min_acc=MIN_ACC):
    """Smallest unique confidence t with accuracy(correct | conf >= t) >= min_acc; None if no t exists."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    for t in np.unique(conf):                      # ascending, so the first hit is the smallest
        if correct[conf >= t].mean() >= min_acc:
            return float(t)
    return None


def fold_stats(t, conf, correct):
    """Held-out metrics for one fold; accepted/rejected accuracy are NaN when that set is empty."""
    acc = conf >= t if t is not None else np.zeros(len(conf), bool)
    m = lambda s: float(correct[s].mean()) if s.any() else float("nan")  # noqa: E731
    return dict(n=len(conf), n_acc=int(acc.sum()), acc_acc=m(acc), share=float(acc.mean()), rej_acc=m(~acc),
                all_acc=float(correct.mean()), correct_acc=int(correct[acc].sum()), correct_rej=int(correct[~acc].sum()))


def cell_stats(rs, min_acc, ident):
    """Per-fold held-out stats of one (source, domain, method, target) cell; rs = its per-fold raw arrays."""
    out = []
    for x in rs:
        t = select_threshold(x["ctr"], x["oktr"], min_acc)
        s = fold_stats(t, x["cva"], x["okva"])
        s.update(**ident, fold=x["f"]["g"], t=t, n_train=len(x["ctr"]),
                 train_acc_accepted=float(x["oktr"][x["ctr"] >= t].mean()) if t is not None else float("nan"))
        out.append(s)
    return out


def cell_summary(st, ident):
    """Aggregate one cell over its folds (decision rule of targets.md: fold mean over folds that accepted something)."""
    d = pd.DataFrame(st)
    ok = d.n_acc > 0
    r = dict(source=ident["src"], domain=ident["domain"], method=ident["method"], target=ident["target"], n_folds=len(d),
             n_folds_no_acceptance=int((~ok).sum()),
             acc_acc_fold_mean=float(d.acc_acc[ok].mean()) if ok.any() else float("nan"),
             acc_acc_fold_min=float(d.acc_acc[ok].min()) if ok.any() else float("nan"),
             acc_acc_fold_max=float(d.acc_acc[ok].max()) if ok.any() else float("nan"),
             acc_acc_pooled=float(d.correct_acc.sum() / d.n_acc.sum()) if ok.any() else float("nan"),
             share_fold_mean=float(d.share.mean()), share_pooled=float(d.n_acc.sum() / d.n.sum()),
             rej_acc_fold_mean=float(d.rej_acc.mean()) if d.rej_acc.notna().any() else float("nan"),
             overall_acc=float(d.correct_acc.add(d.correct_rej).sum() / d.n.sum()),
             t_mean=float(d.t.mean()) if d.t.notna().any() else float("nan"),
             t_min=float(d.t.min()) if d.t.notna().any() else float("nan"),
             t_max=float(d.t.max()) if d.t.notna().any() else float("nan"))
    r["pass"] = bool(ok.any() and r["acc_acc_fold_mean"] >= PASS_ACC)
    return r


def criterion(s):
    """Criterion counts and verdict for the rows `s` of one (source, target); the two aggregation rules are both reported."""
    p = s[s["pass"]]
    out = {"cells_pass": int(len(p)),
           "passing_cells_share_mean": float(p.share_fold_mean.mean()) if len(p) else float("nan"),
           "passing_cells_share_lt_0.20": int((p.share_fold_mean < NARROW_SHARE).sum())}
    # 사전 기준은 "수용 없음" fold 의 집계를 정하지 않았다(검수 E3, targets "검수 반영 개정 2").
    # 그 fold 를 평균에서 뺀 칸 수(cells_pass)와, 그런 fold 가 있는 칸을 통과에서 뺀 칸 수를 둘 다 내고
    # 두 규칙이 7칸 기준에서 엇갈리면 판정은 "split" 이다(한쪽을 결과를 본 뒤 고르지 않는다).
    strict = int((p.n_folds_no_acceptance == 0).sum())
    out["cells_pass_all_folds_accept"] = strict
    lenient = ("fail" if len(p) < PASS_CELLS else
               "holds but narrow" if p.share_fold_mean.mean() < NARROW_SHARE else "holds")
    out["verdict"] = lenient if (len(p) >= PASS_CELLS) == (strict >= PASS_CELLS) else "split"
    return out


def acc_at_share(conf, correct, q):
    """Accuracy of the ceil(q*n) most confident images (stable order, so ties keep image order)."""
    k = max(int(np.ceil(q * len(conf))), 1)
    return float(correct[np.argsort(-conf, kind="stable")[:k]].mean())


def aurc(conf, correct):
    """Area under the risk-coverage curve: mean error rate of the top-q most confident images over q = 0.01..1 (the E risk_coverage grid)."""
    err = (~correct)[np.argsort(-conf, kind="stable")].astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk[np.maximum((GRID * len(err)).astype(int) - 1, 0)].mean())


def error_structure(rs):
    """Pooled over a cell's folds: share of errors that are one level off, levels present, and two majority baselines
    (most frequent held-out level; most frequent TRAINING level applied to the held-out workers)."""
    p, y = np.concatenate([x["pva"] for x in rs]), np.concatenate([x["yva"] for x in rs])
    wrong = p != y
    base = np.concatenate([x["yva"] == np.bincount(x["ytr"]).argmax() for x in rs])
    return dict(adjacent_error_share=float((np.abs(p - y)[wrong] == 1).mean()) if wrong.any() else float("nan"),
                n_levels_present=int(len(np.unique(y))), majority_share_heldout=float(np.bincount(y).max() / len(y)),
                majority_train_level_acc=float(base.mean()), mode_acc=float((~wrong).mean()))


def explor_selective(raw, S, metrics):
    """A2 (curve, AURC, acceptance coverage, error structure, worker bootstrap) and A3 (training-accuracy target sweep)."""
    wk_all = E.G["WK"]
    fold_rows, cell_rows, boot, perw, sweep, sweep_cells = [], [], [], [], [], []
    for (src, dom, m, tg), rs in raw.items():
        ident = dict(src=src, domain=dom, method=m.upper(), target=tg)
        for x in rs:
            r = dict(**ident, fold=x["f"]["g"], n_val=len(x["cva"]), aurc=aurc(x["cva"], x["okva"]), aurc_random=float(1 - x["okva"].mean()),
                     acc_random=float(x["okva"].mean()))
            r.update({f"acc_at_{q}": acc_at_share(x["cva"], x["okva"], q) for q in SHARES})
            fold_rows.append(r)
        sm = S[(S.source == src) & (S.domain == dom) & (S.method == m.upper()) & (S.target == tg)].iloc[0]
        cell_rows.append(dict(**ident, n_folds=int(sm.n_folds), folds_with_acceptance=int(sm.n_folds - sm.n_folds_no_acceptance),
                              acc_acc_worst_fold=sm.acc_acc_fold_min, unaccepted_share=1 - sm.share_pooled, **error_structure(rs)))
        # worker bootstrap of the main (0.96) decision: accepted accuracy and accepted share
        st = cell_stats(rs, MIN_ACC, ident)
        acc = [(x["cva"] >= f["t"]).astype(float) if f["t"] is not None else np.zeros(len(x["cva"])) for x, f in zip(rs, st)]

        def stat(a, ok):
            return np.array([(a * ok).sum() / a.sum() if a.sum() > 0 else np.nan, a.mean()])
        b, w = E.explor_block(dict(analysis=f"selective_{src}", domain=dom, method=m.upper(), target=tg), dom, [x["f"]["val"] for x in rs],
                              wk_all, dict(a=acc, ok=[x["okva"].astype(float) for x in rs]), stat, ["accepted_acc", "accepted_share"])
        boot += b
        perw += w
    # A3: other training-accuracy targets, both aggregation rules
    for ma in MIN_ACC_SWEEP:
        rows = []
        for (src, dom, m, tg), rs in raw.items():
            ident = dict(src=src, domain=dom, method=m.upper(), target=tg)
            r = cell_summary(cell_stats(rs, ma, ident), ident)
            rows.append(r)
            sweep_cells.append(dict(min_acc=ma, **r))
        D = pd.DataFrame(rows)
        for (src, tg), s in D.groupby(["source", "target"]):
            c = criterion(s)
            sweep.append(dict(min_acc=ma, source=src, target=tg, **c, share_fold_mean_all_cells=float(s.share_fold_mean.mean())))
    sw = pd.DataFrame(sweep)
    # consistency: the 0.96 sweep row must equal the main criterion
    for r in sweep:
        if r["min_acc"] == MIN_ACC:
            main = criterion(S[(S.source == r["source"]) & (S.target == r["target"])])
            assert all(r[k] == v or (pd.isna(v) and pd.isna(r[k])) for k, v in main.items()), "sweep 0.96 != main"
    fr, cr = pd.DataFrame(fold_rows), pd.DataFrame(cell_rows)
    fr.to_csv(OUT / "explor_fold_curves.csv", index=False)
    cr.to_csv(OUT / "explor_cells.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT / "explor_boot.csv", index=False)
    pd.DataFrame(perw).to_csv(OUT / "explor_per_worker.csv", index=False)
    sw.to_csv(OUT / "explor_minacc.csv", index=False)
    pd.DataFrame(sweep_cells).to_csv(OUT / "explor_minacc_cells.csv", index=False)
    E.put_metrics(metrics, boot, ("analysis", "domain", "method", "target", "basis", "stat"))
    for (src, dom, m, tg), g in fr.groupby(["src", "domain", "method", "target"]):
        for k in [c for c in fr.columns if c.startswith(("acc_at_", "aurc", "acc_random"))]:
            metrics[f"explor/curve/{src}/{dom}/{m}/{tg}/{k}"] = float(g[k].mean())    # fold mean
    for r in cell_rows:
        for k in ("folds_with_acceptance", "acc_acc_worst_fold", "unaccepted_share", "adjacent_error_share", "n_levels_present",
                  "majority_share_heldout", "majority_train_level_acc"):
            metrics[f"explor/cell/{r['src']}/{r['domain']}/{r['method']}/{r['target']}/{k}"] = r[k]
    # cells where every fold accepts something AND the mean acceptance share is at least NARROW_SHARE
    for (src, tg), s in S.groupby(["source", "target"]):
        c = s[(s.n_folds_no_acceptance == 0) & (s.share_fold_mean >= NARROW_SHARE)]
        k = f"explor/full_accept_share_ge_0.20/{src}/{tg}"
        metrics[f"{k}/n_cells"] = int(len(c))
        metrics[f"{k}/cells"] = ";".join(f"{d}/{m}" for d, m in zip(c.domain, c.method))
        metrics[f"{k}/acc_min"] = float(c.acc_acc_fold_mean.min()) if len(c) else float("nan")
        metrics[f"{k}/acc_max"] = float(c.acc_acc_fold_mean.max()) if len(c) else float("nan")
    for r in sweep:
        for k in ("cells_pass", "cells_pass_all_folds_accept", "passing_cells_share_mean", "share_fold_mean_all_cells", "verdict"):
            metrics[f"explor/minacc/{r['min_acc']}/{r['source']}/{r['target']}/{k}"] = r[k]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = X.build_cohort()
    assert len(coh) == 5889, len(coh)
    smoke = int(os.environ.get("SMOKE", "0"))        # smoke test: random subsample of this many images
    if smoke:
        coh = coh.sample(smoke, random_state=0).reset_index(drop=True)
    N = len(coh)
    ann, _ = E.prepare(coh)
    folds, fw = E.build_folds(coh, X._load_exp012())
    fw.to_csv(OUT / "folds_workers.csv", index=False)
    for f in folds:
        assert not (set(G["WK"][f["train"]]) & set(G["WK"][f["val"]])), "train/eval workers overlap"
        assert not (set(G["WK"][f["donors"]]) & set(G["WK"][f["val"]])), "donor/eval workers overlap"
    print(f"PWW_PROGRESS 0/3 prepared N={N} folds={len(folds)}", flush=True)
    with get_context("fork").Pool(12) as pool:
        gauss = np.zeros((N, E.NCOUNT), np.int16)
        for i, c in pool.imap_unordered(E._w_gauss, range(N), chunksize=20):
            gauss[i] = c
        print("PWW_PROGRESS 1/3 gauss MC", flush=True)
        tasks = [(f["g"], int(i), md) for f in folds for md, ix in ((0, f["val"]), (2, f["train"])) for i in ix]
        emp = {}
        for n, (key, c, _, _) in enumerate(pool.imap_unordered(E._w_emp, tasks, chunksize=40)):
            emp[key] = c
            if n % 5000 == 0:
                print(f"PWW_PROGRESS {n}/{len(tasks)} empirical MC", flush=True)
    print("PWW_PROGRESS 2/3 empirical MC", flush=True)

    rows, per_fold, raw = [], [], {}
    for src in ("A1", "gauss"):
        for dom in E.DOMAINS:
            fl = [f for f in folds if f["domain"] == dom]
            for m in E.METHODS:
                for tg in E.TARGETS:
                    field = E.SPEC[(m, tg)]
                    tv = ann[:, field]
                    rs = []
                    for f in fl:
                        def conf_ok(ix, md):
                            cnt = gauss[ix] if src == "gauss" else np.array([emp[(f["g"], int(i), md)] for i in ix])
                            P = E.probs_of(cnt, field)
                            y = E.y_index(tv[ix], field)
                            return P.max(1), P.argmax(1) == y, P.argmax(1), y
                        ctr, oktr, _, ytr = conf_ok(f["train"], 2)
                        cva, okva, pva, yva = conf_ok(f["val"], 0)
                        rs.append(dict(f=f, ctr=ctr, oktr=oktr, ytr=ytr, cva=cva, okva=okva, pva=pva, yva=yva))
                    raw[(src, dom, m, tg)] = rs
                    ident = dict(src=src, domain=dom, method=m.upper(), target=tg)
                    st = cell_stats(rs, MIN_ACC, ident)
                    per_fold += st
                    rows.append(cell_summary(st, ident))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT / "per_fold.csv", index=False)

    metrics = {"n_images": N, "n_folds": len(folds), "leakage_check/train_eval_worker_overlap": 0}
    for src in ("A1", "gauss"):
        for tg in E.TARGETS:
            metrics.update({f"criterion/{src}/{tg}/{k}": v for k, v in criterion(S[(S.source == src) & (S.target == tg)]).items()})
    for r in rows:
        pre = f"{r['source']}/{r['domain']}/{r['method']}/{r['target']}"
        for k in ("acc_acc_fold_mean", "share_fold_mean", "n_folds_no_acceptance", "pass"):
            metrics[f"{pre}/{k}"] = r[k]
    explor_selective(raw, S, metrics)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("PWW_PROGRESS 3/3 done", flush=True)
    print(S[S.target == "action"].round(3).to_string())


if __name__ == "__main__":
    main()
