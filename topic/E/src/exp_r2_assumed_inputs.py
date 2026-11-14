# PWW-DISCLOSURE: public
"""exp-r2-assumed-inputs: which scorer inputs are observed, assumed or caller-supplied, and how
much the two unobservable ones (wrist posture, external load) move the cohort's scores and
action levels.

Design (paper/targets/targets.md): deterministic re-scoring of the 6,007 cached poses. The
baseline is the main-cohort setting (rescore_batch.py: cohort_context, posture only); it must
reproduce the cached summary.csv before any sensitivity condition is read. Each condition
changes one assumption; the scorers are the existing ergo_v2 ones (no table is re-implemented).

Outputs ($PWW_RESULT_DIR): metrics.json, summary_table.csv, input_inventory.csv.
"""
import json
import os
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ace_pose.batch_infer import ergo_columns  # noqa: E402
from pww_data import resolve  # noqa: E402
from ace_pose.ergo_v2 import pipeline  # noqa: E402
from ace_pose.ergo_v2.context_policy import cohort_context, task_context  # noqa: E402
from ace_pose.major_joints import Major25Pose  # noqa: E402

BATCH = Path(resolve("cohort_pose_cache"))
OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "topic/E/result/exp-r2-assumed-inputs"))
WU = np.array([0.0, -1.0, 0.0])
METHODS = ("OWAS", "RULA", "REBA")
COLS = {"OWAS": ("owas_action", "owas_action"),   # OWAS has no grand score: the action category is the score
        "RULA": ("rula_grand", "rula_action"),
        "REBA": ("reba_grand", "reba_action")}

# Intent: each condition = (name, methods it is read for, score_all kwargs added to the cohort context,
# per-method wrist overrides). Kg values are one representative inside each range the code defines:
#   OWAS load code (owas.py:134): <10 kg =1, 10-20 =2, >20 =3
#   RULA force/load score (rula.py:97): <2 kg 0; 2-10 kg 1 (2 if static/repeated); >10 kg 2 (3 if static/repeated); shock 3
#   REBA load score (reba.py:98): <5 kg 0; 5-10 kg 1; >10 kg 2; shock +1
CONDITIONS = [
    ("wrist_default2", ("RULA", "REBA"), {}, {"RULA": dict(wrist_score_override=2), "REBA": dict(wrist_score_override=2)}),
    # worst = highest wrist sub-score the code allows (RULA 4 with twist 2, REBA 3)
    ("wrist_worst", ("RULA", "REBA"), {}, {"RULA": dict(wrist_score_override=4, wrist_twist_score_override=2),
                                           "REBA": dict(wrist_score_override=3)}),
    ("load_OWAS_code2_15kg", ("OWAS",), dict(load_kg=15.0), {}),
    ("load_OWAS_code3_25kg", ("OWAS",), dict(load_kg=25.0), {}),
    ("load_RULA_force1_5kg", ("RULA",), dict(load_kg=5.0), {}),
    ("load_RULA_force2_5kg_repeated", ("RULA",), dict(load_kg=5.0, load_repeated=True), {}),
    ("load_RULA_force3_15kg_repeated", ("RULA",), dict(load_kg=15.0, load_repeated=True), {}),
    ("load_REBA_1_7.5kg", ("REBA",), dict(load_kg=7.5), {}),
    ("load_REBA_2_15kg", ("REBA",), dict(load_kg=15.0), {}),
    ("load_REBA_3_15kg_shock", ("REBA",), dict(load_kg=15.0, shock=True), {}),
    ("taskload_provisional", METHODS, "TASK", {}),   # task_context(): defined for care/meal-assist tasks only
]


@contextmanager
def wrist_overrides(over):
    """Route the pipeline's RULA/REBA calls through partials that fix the wrist sub-score, so the
    pipeline's own worse-side logic is reused instead of copied."""
    orig = pipeline.score_rula, pipeline.score_reba
    try:
        if "RULA" in over:
            pipeline.score_rula = partial(orig[0], **over["RULA"])
        if "REBA" in over:
            pipeline.score_reba = partial(orig[1], **over["REBA"])
        yield
    finally:
        pipeline.score_rula, pipeline.score_reba = orig


def load_poses(df):
    out = []
    for r in df.itertuples():
        d = np.load(BATCH / "details" / f"{r.file_stem}.npz")
        pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                           confidence=d["conf_m25"].astype(np.float64), source=np.zeros(25, dtype=np.int8))
        pid = getattr(r, "process_id", None)
        out.append((pose, r.process_label, r.domain, None if pd.isna(pid) else pid))
    return out


def run(poses, ctx_fn, extra=None, over=None):
    """Score every pose; ctx_fn(label) gives the context, extra overrides it. Returns a DataFrame of ergo columns."""
    rows = []
    with wrist_overrides(over or {}):
        for pose, label, dom, pid in poses:
            kw = {**ctx_fn(label), **(extra or {})}
            res = pipeline.score_all(pose, world_up=WU, walking=False, **kw, domain=dom, process_id=pid, process_label=label)
            rows.append(ergo_columns(res))
    return pd.DataFrame(rows)


def inventory():
    """Input inventory read from the scorer code (ergo_v2). Cohort value = what rescore_batch.py passes."""
    A = "ace_pose/ergo_v2/"
    O, C, S = "observed", "assumed", "caller-supplied"
    rows = []
    def add(m, name, cls, val, loc): rows.append(dict(method=m, input=name, cls=cls, cohort_value=val, location=loc))
    for m, f in (("OWAS", "owas.py"),):
        add(m, "trunk flexion/twist/lateral (back code)", O, "computed from shoulders/hips; z dropped (2D image plane)", f"{A}{f}:115-117")
        add(m, "arms above shoulder (arm code)", O, "computed from wrists/shoulders", f"{A}{f}:120")
        add(m, "knee flexion, balance, kneeling, sitting (leg code)", O, "computed from hips/knees/ankles", f"{A}{f}:123-127")
        add(m, "walking (leg code 7)", S, "False (rescore_batch.py:46)", f"{A}{f}:97")
        add(m, "load_kg (load code)", S, "0.0 kg -> code 1 (context_policy.py:36)", f"{A}{f}:96,134-139")
        add(m, "world_up", S, "[0,-1,0], image up-axis (rescore_batch.py:45)", f"{A}{f}:98")
    for m, f in (("RULA", "rula.py"), ("REBA", "reba.py")):
        ln = (lambda a, b: a if m == "RULA" else b)
        add(m, "upper-arm flexion and abduction", O, "computed from shoulder/elbow/hip", f"{A}{f}:{ln(131,130)}-{ln(132,131)}")
        add(m, "lower-arm (elbow) flexion", O, "computed from shoulder/elbow/wrist", f"{A}{f}:{ln(133,132)}")
        if m == "RULA":
            add(m, "forearm across midline / out to side (+1)", O, "computed", f"{A}rula.py:145")
        add(m, "wrist flexion", C, "not computable: wrist_flexion_deg() always returns None (no finger keypoints), so wrist = 1 (neutral)", f"{A}angles.py:372-375,367-369;{A}{f}:{ln(149,168)}")
        add(m, "trunk flexion, twist, lateral", O, "computed from shoulders/hips; z dropped", f"{A}{f}:{ln(163,136)}")
        add(m, "neck flexion, twist, lateral", O, "computed from ears (else eyes)/shoulders", f"{A}{f}:{ln(166,139)}")
        add(m, "legs (balance, knee flexion, sitting)", O, "computed; legs_supported=None means auto", f"{A}{f}:{ln(180,153)}")
        add(m, "legs_supported override", S, "None (auto, observed)", f"{A}pipeline.py:26")
        add(m, "shoulder_raised (+1)", C, "False", f"{A}{f}:{ln(138,161)}")
        add(m, "upper_arm_supported (-1)", C, "False", f"{A}{f}:{ln(140,163)}")
        add(m, "wrist_deviation (+1)", C, "False", f"{A}{f}:{ln(150,169)}")
        add(m, "load_kg (force/load score)", S, "0.0 kg (context_policy.py:36)", f"{A}{f}:{ln(158,156)}")
        add(m, "shock", S, "False", f"{A}{f}:{ln(158,156)}")
        add(m, "world_up", S, "[0,-1,0]", f"{A}{f}:{ln(163,136)}")
    add("RULA", "wrist twist", C, "1 (WRIST_TWIST_DEFAULT_RULA); wrist_twist_at_end False", f"{A}rula.py:153-154;{A}angles.py:368")
    add("RULA", "static_minute / repetitive_4_per_min (muscle use)", S, "False / False (task_context static_or_repeated=false)", f"{A}rula.py:157")
    add("RULA", "load_repeated (force/load)", S, "False (context_policy.py:36)", f"{A}rula.py:158")
    add("REBA", "coupling", C, "1 (Fair), pipeline default", f"{A}reba.py:173;{A}pipeline.py:25")
    add("REBA", "activity: static / repetitive / rapid", S, "False / False / False (static from task_context)", f"{A}reba.py:177;{A}pipeline.py:73-75")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BATCH / "summary.csv")
    df = df[df.status == "ok"].reset_index(drop=True)
    poses = load_poses(df)

    # Baseline gate: main-cohort setting must reproduce the cached columns exactly.
    base = run(poses, cohort_context)
    metrics, match = {}, {}
    for col in ("owas_action", "rula_grand", "rula_action", "reba_grand", "reba_action"):
        match[col] = int((base[col].to_numpy() == df[col].to_numpy()).sum())
    allmatch = int(np.all([base[c].to_numpy() == df[c].to_numpy() for c in match], axis=0).sum())
    metrics["baseline/n"] = len(df)
    metrics["baseline/match_all_columns"] = allmatch
    for c, v in match.items():
        metrics[f"baseline/match/{c}"] = v
    print("baseline match", allmatch, "/", len(df), match, flush=True)
    if allmatch != len(df):
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
        sys.exit("baseline does not reproduce cached scores; stopping before the sensitivity conditions")

    rows = []
    for name, methods, extra, over in CONDITIONS:
        res = run(poses, cohort_context, None if extra == "TASK" else extra, over) if extra != "TASK" \
            else run(poses, task_context)
        for m in methods:
            sc, ac = COLS[m]
            for dom in ["ALL", *sorted(df.domain.unique())]:
                k = np.ones(len(df), bool) if dom == "ALL" else (df.domain == dom).to_numpy()
                b_s, b_a, n_s, n_a = (base[sc].to_numpy()[k], base[ac].to_numpy()[k],
                                      res[sc].to_numpy()[k], res[ac].to_numpy()[k])
                dist = {int(l): round(float((n_a == l).mean()), 4) for l in sorted(set(base[ac]) | set(res[ac]))}
                row = dict(assumption=name, method=m, domain=dom, n=int(k.sum()),
                           share_score_changed=float((n_s != b_s).mean()),
                           share_action_changed=float((n_a != b_a).mean()),
                           share_action_up=float((n_a > b_a).mean()),
                           baseline_action_dist=json.dumps({int(l): round(float((b_a == l).mean()), 4) for l in dist}),
                           action_dist=json.dumps(dist))
                rows.append(row)
                for key in ("n", "share_score_changed", "share_action_changed", "share_action_up"):
                    metrics[f"{name}/{m}/{dom}/{key}"] = row[key]
    # wrist_neutral is the baseline itself (changes 0 by construction) and is not re-listed.
    pd.DataFrame(rows).to_csv(OUT / "summary_table.csv", index=False)
    inv = inventory()
    inv.to_csv(OUT / "input_inventory.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("done", len(rows), "rows")


if __name__ == "__main__":
    main()
