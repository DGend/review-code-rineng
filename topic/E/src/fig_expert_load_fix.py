# PWW-DISCLOSURE: public
"""Re-experiment for Section 4.9, decomposed into two code fixes:

  stage 0 "submitted": cached batch_v1 action levels (what the expert saw).
  stage 1 "tables":    same cached pose, re-scored with the current
                       standard-verified ergo_v2 tables (the cohort cache
                       predates the 2026-08-11 tables.py correction).
  stage 2 "legs":      stage 1 plus the 2026-10-05 legs fix (bilateral support
                       assumed instead of the ankle-symmetry heuristic).
  stage 3 "+contact":  stage 2 plus the restored second-person detections,
                       treating wrist contact with another person as an
                       assistive >=20 kg load.
  stage 4 "+worker":   stage 3 on the corrected cohort pose: the analysed
                       person is now matched to the labelled worker in 2D
                       image coordinates (2026-10-07 fix); before it, three
                       of the 22 images (M1, M4, M5) scored a different person.

  stage 5 "+angles":   stage 4 with the image-plane angle scorer and the rule
                       fixes of 2026-10-08 morning (angles.py, rula.py, reba.py).
  "angles_nocontact":  stage 5 without the contact rule.
  stage 6 "+posture":  posture rules corrected after the expert comparison
                       (2026-10-08): arm direction on a deeply bent trunk and
                       overhead arms, neck from the ear-eye line, seated legs
                       and a single bent knee; RULA wrist default still 2.
  stage 7 "+wrist":    stage 6 with an unseen wrist taken as neutral (1).
  stage 8 "+taskload": stage 7 with the task-level load of
                       standards/task_context.json replacing the contact rule;
                       the code the cohort-level results use.

Stages 1-5 were computed with earlier code and are read from
stages_previous_angle_code.csv (that code is no longer in the package);
stages 6-8 are computed here with the current code on the current pose.
and all other scoring arguments keep the batch_v1 defaults, so each stage
differs from the previous one by exactly one change. The expert's own RULA
sub-scores and Table A value are also used to check which Table A (cached or
current) the expert's hand lookup agrees with.

The contact rule is applied uniformly to every eldercare image (domain known a
priori from the F06 site code, not from expert labels); for logistics and
manufacturing it is only reported as a would-fire diagnostic, because a second
person there is a co-worker, not a supported care recipient.

Source data (real, not fabricated):
  pww_data.resolve("expert_pilot_22"): file names, submitted and expert scores
  ace_pose/_runs/batch_v1/details/<stem>.npz (cached primary pose)
  original JPEGs, re-inferred with RTMW3D (rtmlib Wholebody3d, CPU)
Writes:
  ace_pose/_runs/expert_pilot/load_fix_before_after.csv
"""
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.major_joints import Major25Pose, coco_wholebody_to_major25
from ace_pose.ergo_v2.context_policy import task_context
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.vlm_context import HeuristicBackend
from pww_data import resolve

from ace_pose.data_paths import IMG_DIR
DET_DIR = ROOT / "ace_pose/_runs/batch_v1/details"
OLD_DET_DIR = ROOT / "ace_pose/_runs/expert_pilot/details_as_submitted"
# pre-correction cohort scores, kept to document the old Table-A values
COHORT_CSV = ROOT / "ace_pose/_runs/batch_v1/_backup_pre_tablefix_20261004/summary_with_xw.csv"
STAGES = ("submitted", "tables", "legs", "contact", "worker", "angles", "posture", "wrist", "taskload")
ARCHIVED = ("tables", "legs", "contact", "worker", "angles", "angles_nocontact")
ARCHIVE = ROOT / "ace_pose/_runs/expert_pilot/stages_previous_angle_code.csv"
OUT_CSV = ROOT / "ace_pose/_runs/expert_pilot/load_fix_before_after.csv"
WORLD_UP = np.array([0.0, -1.0, 0.0])

def to_2d_major25(kpts2d_133, scores_133):
    padded = np.concatenate([kpts2d_133, np.zeros((133, 1))], axis=1)
    return coco_wholebody_to_major25(padded, scores_133).keypoints


def old_legs_balanced(pose, frame):
    # pre-2026-10-05 ankle-symmetry heuristic, kept only to isolate the "tables" stage
    from ace_pose.major_joints import ANKLE_L, ANKLE_R, PELVIS
    k = pose.keypoints
    al = np.dot(k[ANKLE_L] - k[PELVIS], frame.side)
    ar = np.dot(k[ANKLE_R] - k[PELVIS], frame.side)
    return (al < 0) != (ar < 0) and abs(abs(al) - abs(ar)) < 0.4 * max(abs(al), abs(ar), 1e-6)


def actions_old_legs(pose):
    import ace_pose.ergo_v2.owas as O, ace_pose.ergo_v2.rula as RU, ace_pose.ergo_v2.reba as RE
    saved = O.legs_balanced
    O.legs_balanced = RU.legs_balanced = RE.legs_balanced = old_legs_balanced
    try:
        return actions(pose, 0.0)
    finally:
        O.legs_balanced = RU.legs_balanced = RE.legs_balanced = saved


@contextmanager
def rula_wrist_default(score):
    """Temporarily restore an earlier RULA default wrist score (stage 6 isolates the wrist change)."""
    import ace_pose.ergo_v2.rula as RU
    kw = RU._score_wrist.__kwdefaults__
    saved, kw["default"] = kw["default"], score
    try:
        yield
    finally:
        kw["default"] = saved


def actions(pose, load_kg):
    r = score_all(pose, world_up=WORLD_UP, load_kg=load_kg, walking=False)
    return {"OWAS": r.owas.action_category, "RULA": r.rula.action_level, "REBA": r.reba.action_level}


def same_person(k_old, k_new, tol=0.05):
    """Same analysed person if the body keypoints moved by under 5% of the body size
    (re-inference on another device shifts them by well under 1%)."""
    a, b = k_old[:17], k_new[:17]
    size = max(float(np.hypot(*(b.max(0) - b.min(0)))), 1.0)
    return bool(np.median(np.linalg.norm(a - b, axis=1)) / size < tol)


def main():
    from rtmlib import Wholebody3d
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        model = Wholebody3d(mode="balanced", backend="onnxruntime", device="cpu")
    scores = pd.read_csv(resolve("expert_pilot_22"))
    ids = scores.drop_duplicates("id").set_index("id").file_stem.to_dict()
    labels = pd.read_csv(ROOT / "ace_pose/_runs/batch_v1/summary.csv").set_index("file_stem").process_label
    heuristic = HeuristicBackend()

    rows = []
    def evaluate(id_, img, cached):
        """Scores of the cached pose: old legs rule, legs fix, and legs fix plus contact."""
        pose = Major25Pose(keypoints=cached["kpts_m25"].astype(np.float64),
                           confidence=cached["conf_m25"].astype(np.float64),
                           source=np.zeros(25, dtype=np.int8))
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            k3, sc, _simcc, k2 = model(img)
        # primary = re-inferred person whose major-25 3D pose matches the cached one
        dists = [float(np.nanmean(np.linalg.norm(
            coco_wholebody_to_major25(k3[i], sc[i]).keypoints - pose.keypoints, axis=1)))
            for i in range(k3.shape[0])]
        p = int(np.argmin(dists))
        others = [to_2d_major25(k2[i], sc[i]) for i in range(k3.shape[0]) if i != p]
        ctx = heuristic.extract(pose=pose, worker_metadata={
            "primary_kpts_2d": to_2d_major25(k2[p], sc[p]),
            "other_people_kpts_2d": others,
        })
        contact = ctx.load_kg == HeuristicBackend.CONTACT_LOAD_KG
        applied = contact and id_.startswith("E")
        legs = actions(pose, 0.0)
        return dict(pose=pose, tables=actions_old_legs(pose), legs=legs,
                    contact=actions(pose, HeuristicBackend.CONTACT_LOAD_KG) if applied else legs,
                    fired=contact, applied=applied, n_people=k3.shape[0], err=round(dists[p], 3))

    rows = []
    for id_, stem in ids.items():
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        old = evaluate(id_, img, np.load(OLD_DET_DIR / f"{stem}.npz"))
        new = evaluate(id_, img, np.load(DET_DIR / f"{stem}.npz"))
        with rula_wrist_default(2):
            new_w2 = evaluate(id_, img, np.load(DET_DIR / f"{stem}.npz"))
        r = score_all(new["pose"], world_up=WORLD_UP, walking=False, **task_context(labels[stem]))
        task = {"OWAS": r.owas.action_category, "RULA": r.rula.action_level, "REBA": r.reba.action_level}

        def level(std, rater):
            g = scores[(scores.id == id_) & (scores.standard == std) & (scores.rater == rater)]
            return int(g.action.iloc[0])
        # what the expert actually saw (pre-correction cohort scores)
        submitted = {std: level(std, "system_submitted") for std in ("OWAS", "RULA", "REBA")}
        for std in ("OWAS", "RULA", "REBA"):
            rows.append({"id": id_, "standard": std, "n_people": old["n_people"],
                         "primary_match_err": old["err"], "applied": old["applied"],
                         "submitted": submitted[std], "tables": old["tables"][std], "legs": old["legs"][std],
                         "contact": old["contact"][std], "posture": new_w2["contact"][std],
                         "wrist": new["contact"][std], "taskload": task[std],
                         "expert": level(std, "expert"), "contact_fired": old["fired"],
                         "contact_fired_worker": new["fired"],
                         "same_person": same_person(np.load(OLD_DET_DIR / f"{stem}.npz")["kpts2d_133"],
                                                    np.load(DET_DIR / f"{stem}.npz")["kpts2d_133"])})

    df = pd.DataFrame(rows)
    arc = pd.read_csv(ARCHIVE).set_index(["id", "standard"])
    key = pd.MultiIndex.from_frame(df[["id", "standard"]])
    for st in ARCHIVED:
        df[st] = arc.loc[key, st].to_numpy()
    df.to_csv(OUT_CSV, index=False)

    per_img = df.drop_duplicates("id")[["id", "n_people", "primary_match_err", "contact_fired", "applied",
                                        "contact_fired_worker", "same_person"]]
    print(per_img.to_string(index=False))

    def summary(sub, label):
        print(f"\n=== {label} ===")
        for std in ("OWAS", "RULA", "REBA"):
            s = sub[sub.standard == std]
            n = len(s)
            parts = []
            for st in STAGES + ("angles_nocontact",):
                ex = int((s[st] == s.expert).sum())
                un = int((s[st] < s.expert).sum())
                ov = int((s[st] > s.expert).sum())
                mae = float((s[st] - s.expert).abs().mean())
                parts.append(f"{st}: exact {ex}/{n} under {un} over {ov} MAE {mae:.2f}")
            print(f"{std} | " + " | ".join(parts))

    summary(df[df.id.str.startswith("E")], "eldercare (n=10)")
    summary(df, "all 22 held-out images")

    print("\n=== per-image changes (" + " -> ".join(STAGES) + ", expert) ===")
    for _, r in df.iterrows():
        seq = [r[s] for s in STAGES]
        if len(set(seq)) > 1:
            print(f"{r.id} {r.standard}: {' -> '.join(map(str, seq))} (expert {r.expert})")
    print(f"images whose analysed person changed with the 2026-10-07 fix: {per_img[~per_img.same_person].id.tolist()}")
    fp = per_img[(~per_img.id.str.startswith("E")) & per_img.contact_fired]
    print(f"\ncontact would fire on logistics/manufacturing (not applied): {fp.id.tolist()}")
    print(f"contact fired on eldercare: {per_img[per_img.id.str.startswith('E') & per_img.contact_fired].id.tolist()}")
    print(f"wrote {OUT_CSV}")

    table_a_check(scores)
    leg_subscore_check(scores)


def leg_subscore_check(scores):
    """Leg sub-score agreement with the expert: as submitted vs current code."""
    print("\n=== leg sub-score agreement with the expert (n=22) ===")
    for std, attr in (("OWAS", "owas"), ("REBA", "reba")):
        sub = scores[(scores.standard == std)]
        exp = sub[sub.rater == "expert"].set_index("id").legs
        submitted = sub[sub.rater == "system_submitted"].set_index("id").legs
        current = {}
        for id_, stem in sub.drop_duplicates("id").set_index("id").file_stem.items():
            d = np.load(DET_DIR / f"{stem}.npz")
            pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                               confidence=d["conf_m25"].astype(np.float64),
                               source=np.zeros(25, dtype=np.int8))
            r = score_all(pose, world_up=WORLD_UP)
            current[id_] = getattr(r, attr).legs
        cur = pd.Series(current)
        print(f"{std} legs: submitted {int((submitted == exp).sum())}/22, current code {int((cur == exp).sum())}/22")


def table_a_check(scores):
    """Which Table A does the expert's own hand lookup agree with?"""
    from ace_pose.ergo_v2.tables import RULA_TABLE_A
    cohort = pd.read_csv(COHORT_CSV)
    cohort = cohort[cohort.status == "ok"]
    cached_table = (cohort.groupby(["rula_upper_arm", "rula_lower_arm", "rula_wrist"])
                    .rula_score_a.agg(lambda s: s.mode().iloc[0]).to_dict())
    d = scores[(scores.standard == "RULA") & (scores.rater == "expert")]
    n = agree_cur = 0
    for _, r in d.iterrows():
        ua, la, w, sa = int(r.upper_arm), int(r.lower_arm), int(r.wrist), int(r.score_a)
        n += 1
        agree_cur += int(RULA_TABLE_A[ua - 1, la - 1, w - 1, 0] == sa)
    print(f"\n=== RULA Table A vs expert's own lookup (n={n} expert rows) ===")
    print(f"current verified table agrees with the expert's hand lookup: {agree_cur}/{n}")
    # The cohort always uses the default wrist score 2 while the expert scored a
    # neutral wrist (1), so the expert's cells never occur in the cohort cache;
    # compare the cache against the current table on the cells it does use.
    print("cached cohort Score A vs current table, on cells the cohort uses (wrist twist 1):")
    for (ua, la, w), sa in sorted(cached_table.items()):
        cur = int(RULA_TABLE_A[ua - 1, la - 1, w - 1, 0])
        n_cell = int(((cohort.rula_upper_arm == ua) & (cohort.rula_lower_arm == la)
                      & (cohort.rula_wrist == w)).sum())
        flag = "" if cur == sa else "  <-- differs"
        print(f"  UA{ua} LA{la} W{w}: cached {sa}, current {cur} (n={n_cell}){flag}")


if __name__ == "__main__":
    main()
