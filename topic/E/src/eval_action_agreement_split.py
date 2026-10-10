# PWW-DISCLOSURE: public
"""80/20 split of the label-referenced action-level comparison, with bootstrap CIs.

Nothing in the pipeline is fitted to these labels (frozen pose model, standard tables),
but the 20% evaluation list is frozen here so that any later tuning cannot touch it.
Every evaluated image has an annotation of the analysed worker (eval_action_agreement.py).
The split is random within domain (fixed seed), so each domain keeps its share.

Usage: python topic/E/src/eval_action_agreement_split.py  (after eval_action_agreement.py)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_action_agreement import LEVELS, OUT, metrics  # noqa: E402

SEED, TEST_SHARE, N_BOOT = 20261008, 0.2, 1000


def split(per: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED)
    test = set()
    for _, g in per.groupby("domain"):
        stems = sorted(g.file_stem)
        test |= set(rng.choice(stems, size=round(TEST_SHARE * len(stems)), replace=False))
    return per.file_stem.isin(test)


def boot_ci(ref, sys_, lv, rng):
    keys = ("accuracy", "macro_f1", "weighted_f1", "kappa_quadratic")
    ref, sys_ = np.asarray(ref), np.asarray(sys_)
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(ref), len(ref))
        m = metrics(ref[i], sys_[i], lv)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def main():
    per = pd.read_csv(OUT / "per_image_labels.csv")
    lst = OUT / "test_split_20.csv"
    is_test = split(per)
    if lst.exists():                                   # frozen: must reproduce the stored list
        if set(pd.read_csv(lst).file_stem) != set(per.file_stem[is_test]):
            sys.exit(f"{lst} differs from the seeded split; the evaluation set is frozen")
    else:
        per.loc[is_test, ["file_stem", "domain"]].to_csv(lst, index=False)
        (OUT / "test_split_20.sha256").write_text(hashlib.sha256(lst.read_bytes()).hexdigest() + "  test_split_20.csv\n")
    rng = np.random.default_rng(SEED)
    res = {"n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
           "test_by_domain": per[is_test].domain.value_counts().to_dict()}
    for part, mask in (("test20", is_test), ("train80", ~is_test)):
        sub = per[mask]
        for std, lv in LEVELS.items():
            m = metrics(sub[f"label_{std}"], sub[f"system_{std}"], lv)
            if part == "test20":
                m["ci95"] = boot_ci(sub[f"label_{std}"], sub[f"system_{std}"], lv, rng)
                for dom, g in sub.groupby("domain"):
                    res[f"test20_{dom}_{std}"] = metrics(g[f"label_{std}"], g[f"system_{std}"], lv)
                for v in ("view30", "view45"):
                    g = sub[sub[f"system_{std}_ok_{v}"]]
                    res[f"test20_{v}_{std}"] = metrics(g[f"label_{std}"], g[f"system_{std}"], lv)
            res[f"{part}_{std}"] = m
    (OUT / "metrics_split.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print({k: res[k] for k in ("n_train", "n_test", "test_by_domain")})
    for k, v in res.items():
        if isinstance(v, dict) and "accuracy" in v:
            ci = v.get("ci95")
            s = (f"  CI acc {ci['accuracy'][0]:.2f}-{ci['accuracy'][1]:.2f} mF1 {ci['macro_f1'][0]:.2f}-{ci['macro_f1'][1]:.2f} "
                 f"wF1 {ci['weighted_f1'][0]:.2f}-{ci['weighted_f1'][1]:.2f} kq {ci['kappa_quadratic'][0]:.2f}-{ci['kappa_quadratic'][1]:.2f}") if ci else ""
            print(f"{k:28s} n={v['n']:4d} acc {v['accuracy']:.2f} mF1 {v['macro_f1']:.2f} wF1 {v['weighted_f1']:.2f} "
                  f"k {v['kappa']:.2f} kl {v['kappa_linear']:.2f} kq {v['kappa_quadratic']:.2f} ±1 {v['within_one']:.2f} "
                  f"MAE {v['mae']:.2f} low {v['system_lower']:.2f} high {v['system_higher']:.2f}{s}")


if __name__ == "__main__":
    main()
