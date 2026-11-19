# PWW-DISCLOSURE: public
"""Wrist override must raise (never lower) RULA/REBA scores on a real cached pose, and restore the pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bioera/analysis/src"))
import numpy as np  # noqa: E402
import exp_r2_assumed_inputs as ex  # noqa: E402
from ace_pose.ergo_v2 import pipeline  # noqa: E402
from ace_pose.ergo_v2.context_policy import cohort_context  # noqa: E402
from ace_pose.major_joints import Major25Pose  # noqa: E402


def test_wrist_override_monotone_and_restored():
    d = np.load(next((ex.BATCH / "details").glob("*.npz")))
    pose = Major25Pose(keypoints=d["kpts_m25"].astype(float), confidence=d["conf_m25"].astype(float),
                       source=np.zeros(25, dtype=np.int8))
    orig = pipeline.score_rula
    kw = dict(world_up=ex.WU, walking=False, **cohort_context("x"))
    base = pipeline.score_all(pose, **kw)
    with ex.wrist_overrides({"RULA": dict(wrist_score_override=4), "REBA": dict(wrist_score_override=3)}):
        hi = pipeline.score_all(pose, **kw)
    assert hi.rula.grand_score >= base.rula.grand_score and hi.reba.grand_score >= base.reba.grand_score
    assert pipeline.score_rula is orig
