# PWW-DISCLOSURE: public
"""Confusion matrices and classification metrics for OWAS/RULA/REBA action levels.

Two references:
  expert   the ergonomics expert's action levels on the 22 pilot images (development
           set), for the scores as submitted and for the current code;
  labels   all cohort images: the action level computed from the dataset's annotated
           2D keypoints, against the one computed from the pipeline's pose. Both use the
           same scorer, task context, and joint set (the annotation has no ears, small
           toes, or heels, so those are dropped from the pipeline pose as well), so any
           difference comes from pose estimation alone.
Metrics: accuracy, macro and weighted F1, per-class precision/recall/F1, Cohen's kappa
(unweighted, linear, quadratic), within-one-level rate, mean absolute level difference,
and the share scored lower/higher than the reference.

Usage: python topic/E/src/eval_action_agreement.py   (writes $PWW_RESULT_DIR or
       topic/E/result/action_agreement/: metrics.json, confusion_*.csv, per_image_labels.csv)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ace_pose.data_paths import JSON_DIR  # noqa: E402
from ace_pose.ergo_v2.context_policy import task_context  # noqa: E402
from ace_pose.ergo_v2.measurability import measurable, unmeasurable_items  # noqa: E402
from ace_pose.ergo_v2.pipeline import score_all  # noqa: E402
from ace_pose.major_joints import NECK, PELVIS, Major25Pose  # noqa: E402

OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "topic/E/result/action_agreement"))
BATCH = ROOT / "ace_pose/_runs/batch_v1"
WU = np.array([0.0, -1.0, 0.0])
LEVELS = {"OWAS": [1, 2, 3, 4], "RULA": [1, 2, 3, 4], "REBA": [0, 1, 2, 3, 4]}
# major-25 index -> annotation index (the annotation has no ears, small toes, or heels)
M25_FROM_LABEL = {0: 3, 1: 1, 2: 2, 5: 5, 6: 11, 7: 6, 8: 12, 9: 7, 10: 13, 11: 19, 12: 23,
                  13: 20, 14: 24, 15: 21, 16: 25, 17: 22, 20: 26}
DROPPED = [3, 4, 18, 19, 21, 22]   # ears, small toes, heels


def metrics(ref, sys_, levels):
    ref, sys_ = np.asarray(ref, int), np.asarray(sys_, int)
    p, r, f, n = precision_recall_fscore_support(ref, sys_, labels=levels, zero_division=0)
    present = sorted(set(ref) | set(sys_))   # classes absent from both would add a spurious F1 of 0
    _, _, f_mac, _ = precision_recall_fscore_support(ref, sys_, labels=present, average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(ref, sys_, labels=present, average="weighted", zero_division=0)
    d = sys_ - ref
    kap = lambda w: float(cohen_kappa_score(ref, sys_, labels=levels, weights=w))  # noqa: E731
    return {
        "n": int(len(ref)), "accuracy": float((d == 0).mean()), "macro_f1": float(f_mac), "weighted_f1": float(f_w),
        "kappa": kap(None), "kappa_linear": kap("linear"), "kappa_quadratic": kap("quadratic"),
        "within_one": float((abs(d) <= 1).mean()), "mae": float(abs(d).mean()),
        "system_lower": float((d < 0).mean()), "system_higher": float((d > 0).mean()),
        "per_class": {str(lv): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, lv in enumerate(levels)},
        "confusion_rows_reference_cols_system": confusion_matrix(ref, sys_, labels=levels).tolist(),
    }


VIEWS = {"view30": 0.24, "view45": 0.33}   # side-view limits: within 30 or 45 degrees (measurability.py)


def levels_of(pose, label):
    r = score_all(pose, world_up=WU, walking=False, **task_context(label))
    out = {"OWAS": r.owas.action_category, "RULA": r.rula.action_level, "REBA": r.reba.action_level}
    for v, lim in VIEWS.items():
        items = unmeasurable_items(pose, side_view_max=lim)
        for s in ("OWAS", "RULA", "REBA"):
            out[f"{s}_ok_{v}"] = measurable(items, s)
    return out


def with_neck_pelvis(kp, conf):
    kp, conf = kp.copy(), conf.copy()
    for tgt, (a, b) in ((NECK, (5, 6)), (PELVIS, (11, 12))):
        kp[tgt] = (kp[a] + kp[b]) / 2
        conf[tgt] = min(conf[a], conf[b])
    return Major25Pose(keypoints=kp, confidence=conf, source=np.zeros(25, np.int8))


def label_pose(row):
    kp, conf = np.zeros((25, 3)), np.zeros(25)
    for m, l in M25_FROM_LABEL.items():
        if row[f"keypoint_{l}_v"] > 0:
            kp[m, :2] = row[f"keypoint_{l}_x"], row[f"keypoint_{l}_y"]
            conf[m] = 1.0
    return with_neck_pelvis(kp, conf)


def system_pose(stem, same_joints):
    z = np.load(BATCH / "details" / f"{stem}.npz")
    kp, conf = z["kpts_m25"].astype(float), z["conf_m25"].astype(float)
    if same_joints:
        conf = conf.copy()
        conf[DROPPED] = 0.0
    return Major25Pose(keypoints=kp, confidence=conf, source=np.zeros(25, np.int8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(BATCH / "summary.csv")
    res = {}

    # --- expert, 22 development images (needs the expert's consent to release) ---
    expert_csv = ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv"
    if not expert_csv.is_file():
        print(f"skip expert comparison: {expert_csv.relative_to(ROOT)} not included")
    else:
        expert_metrics(pd.read_csv(expert_csv), summ, res)

    # --- annotated keypoints, all cohort images ---
    labels_csv = Path(JSON_DIR).parent / "data_v5.csv"
    if labels_csv.is_file():
        per = label_levels(pd.read_csv(labels_csv), summ)
        per.to_csv(OUT / "per_image_labels.csv", index=False)
        res["labels_excluded"] = {"cohort_ok": int(((summ.status == "ok") & summ.label_match_ok.astype(bool)).sum()),
                                  "evaluated": int(len(per))}
    elif (OUT / "per_image_labels.csv").is_file():
        # the dataset's annotations are distributed by AI-Hub, not with this code; recompute the
        # metrics from the per-image levels shipped with it
        print("dataset annotations not found; recomputing metrics from per_image_labels.csv")
        per = pd.read_csv(OUT / "per_image_labels.csv")
    else:
        sys.exit("neither the dataset annotations nor per_image_labels.csv is available")
    label_metrics(per, res)
    if expert_csv.is_file():
        expert_annotated(pd.read_csv(expert_csv), per, res)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res)


def expert_metrics(s, summ, res):
    act = s.set_index(["id", "standard", "rater"]).action
    stems = s.drop_duplicates("id").set_index("id").file_stem
    lab_of = summ.set_index("file_stem").process_label
    cur = {i: levels_of(system_pose(st, False), lab_of[st]) for i, st in stems.items()}
    for std, lv in LEVELS.items():
        ref = [act[(i, std, "expert")] for i in stems.index]
        res[f"expert22_submitted_{std}"] = metrics(ref, [act[(i, std, "system_submitted")] for i in stems.index], lv)
        res[f"expert22_current_{std}"] = metrics(ref, [cur[i][std] for i in stems.index], lv)
        for v in VIEWS:
            ok = [i for i in stems.index if cur[i][f"{std}_ok_{v}"]]
            if ok:
                res[f"expert22_current_{v}_{std}"] = metrics([act[(i, std, "expert")] for i in ok], [cur[i][std] for i in ok], lv)



def expert_annotated(s, per, res):
    """The 22 expert images scored from the dataset's annotated keypoints instead of the pose model:
    how far a perfect 2D pose would bring the pipeline toward the expert."""
    stems = s.drop_duplicates("id").set_index("id").file_stem
    act = s.set_index(["id", "standard", "rater"]).action
    lev = per.set_index("file_stem")
    ids = [i for i in stems.index if stems[i] in lev.index]
    for std, lv in LEVELS.items():
        res[f"expert22_annotated_pose_{std}"] = metrics([act[(i, std, "expert")] for i in ids],
                                                       [lev.at[stems[i], f"label_{std}"] for i in ids], lv)


def label_levels(lab, summ):
    lab["file_stem"] = lab.file_name.str[:-4]
    ok = summ[(summ.status == "ok") & summ.label_match_ok.astype(bool)]
    m = ok.merge(lab, left_on=["file_stem", "worker_id"], right_on=["file_stem", "worker"], how="inner")
    m = m.drop_duplicates("file_stem", keep=False)          # one annotation per image and worker
    rows = []
    for r in m.to_dict("records"):
        g = levels_of(label_pose(r), r["process_label"])
        p = levels_of(system_pose(r["file_stem"], True), r["process_label"])
        rows.append({"file_stem": r["file_stem"], "domain": r["domain"], "angle": r["angle"],
                     **{f"label_{k}": g[k] for k in LEVELS}, **{f"system_{k}": v for k, v in p.items()}})
    return pd.DataFrame(rows)


def label_metrics(per, res):
    for std, lv in LEVELS.items():
        res[f"labels_all_{std}"] = metrics(per[f"label_{std}"], per[f"system_{std}"], lv)
        for v in VIEWS:
            sub = per[per[f"system_{std}_ok_{v}"]]
            res[f"labels_{v}_{std}"] = metrics(sub[f"label_{std}"], sub[f"system_{std}"], lv)
        for dom, g in per.groupby("domain"):
            res[f"labels_{dom}_{std}"] = metrics(g[f"label_{std}"], g[f"system_{std}"], lv)
    res["coverage"] = {v: {s: {"all": float(per[f"system_{s}_ok_{v}"].mean()),
                               **{d: float(g[f"system_{s}_ok_{v}"].mean()) for d, g in per.groupby("domain")}}
                           for s in LEVELS} for v in VIEWS}


def report(res):
    print(f"{'set':34s} {'acc':>5s} {'mF1':>5s} {'wF1':>5s} {'k':>5s} {'k_lin':>5s} {'k_quad':>6s} {'±1':>5s} {'MAE':>5s} {'low':>5s} {'high':>5s}")
    for k, v in res.items():
        if "accuracy" in v:
            print(f"{k:34s} {v['accuracy']:5.2f} {v['macro_f1']:5.2f} {v['weighted_f1']:5.2f} {v['kappa']:5.2f} "
                  f"{v['kappa_linear']:5.2f} {v['kappa_quadratic']:6.2f} {v['within_one']:5.2f} {v['mae']:5.2f} "
                  f"{v['system_lower']:5.2f} {v['system_higher']:5.2f}  (n={v['n']})")
    print(res.get("labels_excluded"), "->", OUT)
    print(json.dumps(res["coverage"], indent=1))


if __name__ == "__main__":
    main()
