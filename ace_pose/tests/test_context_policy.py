"""Task-level context for the modifiers a single frame cannot show (2026-10-08).

Scoring every pose with zero load made RULA and REBA systematically lower than the
expert, whose scores add force/load for the handled load of the task. The expert
did not add muscle use or activity for posture held from one frame, so the policy
does not either.
"""
from ace_pose.ergo_v2.context_policy import task_context
from ace_pose.ergo_v2.pipeline import score_all
from ace_pose.tests.test_ergo_v2_golden import WORLD_UP, _upright_skeleton


def test_posture_from_one_frame_is_not_taken_as_static():
    assert task_context(None) == {"static_minute": False, "load_kg": 0.0, "load_repeated": False}
    assert task_context("unlisted task")["load_kg"] == 0.0


def test_patient_transfer_carries_more_than_20_kg_repeatedly():
    c = task_context("이동")
    assert c["load_kg"] > 20.0 and c["load_repeated"] and not c["static_minute"]


def test_repeated_load_raises_rula_force_but_not_muscle_use():
    pose = _upright_skeleton(trunk_flex_deg=30.0)
    bare = score_all(pose, world_up=WORLD_UP)
    ctx = score_all(pose, world_up=WORLD_UP, **task_context("이동"))
    assert ctx.rula.load_a == 3 and not ctx.rula.muscle_use_a
    assert ctx.reba.score_a == ctx.reba.posture_a + 2 and ctx.reba.activity_score == 0
    assert ctx.owas.load == 3
    assert ctx.rula.grand_score > bare.rula.grand_score and ctx.reba.grand_score > bare.reba.grand_score


def test_cohort_context_scores_posture_only():
    from ace_pose.ergo_v2.context_policy import cohort_context
    # task loads exist only for some tasks (provisional), so cohort-wide comparisons leave them out
    assert cohort_context("이동") == {"static_minute": False, "load_kg": 0.0, "load_repeated": False}
    assert cohort_context(None) == cohort_context("unlisted task")
