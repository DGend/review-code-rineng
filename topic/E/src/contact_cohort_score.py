# PWW-DISCLOSURE: public
"""Cohort-wide sensitivity analysis for the second-person contact-load rule
(ace_pose.vlm_context.HeuristicBackend, Section 4.9).

For every cohort pose, uses the cached multi-person re-inference
(contact_cohort_infer.py) to decide whether the worker's wrist touches another
person, and reports how often the rule fires per domain. As in the pilot, the
assistive load is applied to eldercare only (a second person there is a care
recipient); for logistics and manufacturing the firing rate is reported as a
would-fire diagnostic. Main-text cohort results do not use this rule.

Source data: ace_pose/_runs/contact_cohort/cache/*.npz, batch_v1 summary/details
Writes ace_pose/_runs/contact_cohort/contact_cohort_scores.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.ergo_v2.context_policy import cohort_context  # posture only, see context_policy
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.major_joints import Major25Pose
from ace_pose.vlm_context import HeuristicBackend

BATCH = ROOT / "ace_pose/_runs/batch_v1"
CACHE = ROOT / "ace_pose/_runs/contact_cohort/cache"
OUT = ROOT / "ace_pose/_runs/contact_cohort/contact_cohort_scores.csv"
UP = np.array([0.0, -1.0, 0.0])


def main():
    df = pd.read_csv(BATCH / "summary.csv")
    df = df[df.status == "ok"]
    hb = HeuristicBackend()
    rows = []
    for r in df.itertuples():
        c = np.load(CACHE / f"{r.file_stem}.npz")
        d = np.load(BATCH / "details" / f"{r.file_stem}.npz")
        pose = Major25Pose(keypoints=d["kpts_m25"].astype(np.float64),
                           confidence=d["conf_m25"].astype(np.float64),
                           source=np.zeros(25, dtype=np.int8))
        ctx = hb.extract(pose=pose, worker_metadata={
            "primary_kpts_2d": c["p2d"], "other_people_kpts_2d": list(c["others2d"])})
        fired = ctx.load_kg == HeuristicBackend.CONTACT_LOAD_KG
        row = {"file_stem": r.file_stem, "domain": r.domain, "n_people": int(c["n_people"]),
               "fired": fired, "owas": r.owas_action, "rula": r.rula_action, "reba": r.reba_action}
        if fired and r.domain == "eldercare":
            tc = cohort_context(r.process_label)
            x = score_all(pose, world_up=UP, static_minute=tc["static_minute"],
                          load_kg=max(tc["load_kg"], HeuristicBackend.CONTACT_LOAD_KG))
            row.update(owas_c=x.owas.action_category, rula_c=x.rula.action_level, reba_c=x.reba.action_level)
        else:
            row.update(owas_c=r.owas_action, rula_c=r.rula_action, reba_c=r.reba_action)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    for dom, g in out.groupby("domain"):
        multi = (g.n_people > 1).mean()
        print(f"{dom:14s} n={len(g):5d}  >1 person detected {100*multi:5.1f}%  "
              f"contact fires {int(g.fired.sum()):4d} ({100*g.fired.mean():.1f}%)")
    e = out[out.domain == "eldercare"]
    print("\neldercare, rule applied (action level changed / moved up):")
    for m in ("owas", "rula", "reba"):
        ch = (e[m] != e[f"{m}_c"])
        print(f"  {m.upper()}: {int(ch.sum())} of {len(e)} poses changed, "
              f"mean level {e[m].mean():.2f} -> {e[f'{m}_c'].mean():.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
