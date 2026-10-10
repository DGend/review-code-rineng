"""Tests for `ace_pose.vlm_context` (W - VLM context extraction)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.vlm_context import (
    COUPLING_TO_INT, ContextExtractor, HeuristicBackend, LOAD_BAND_TO_KG,
    VLMContext, context_to_columns, parse_vlm_json,
)
from ace_pose.tests.test_ergo_v2_golden import _upright_skeleton


class TestParsing:
    def test_clean_json(self):
        text = ('{"load_kg_band": "5-10", "coupling": "fair", '
                '"activity_static": true, "activity_repetitive": false, '
                '"activity_rapid": false, "confidence": 0.7}')
        ctx = parse_vlm_json(text)
        assert ctx.load_band == "5-10"
        assert ctx.load_kg == LOAD_BAND_TO_KG["5-10"]
        assert ctx.coupling == COUPLING_TO_INT["fair"]
        assert ctx.activity_static is True
        assert ctx.activity_repetitive is False
        assert ctx.confidence == pytest.approx(0.7)

    def test_messy_prefix_and_suffix(self):
        text = ('Sure! Here is the JSON: {"load_kg_band": "<5", '
                '"coupling": "good", "activity_static": false} -- end')
        ctx = parse_vlm_json(text)
        assert ctx.load_band == "<5"
        assert ctx.coupling == 0
        assert ctx.activity_static is False

    def test_unknown_band_falls_back(self):
        text = '{"load_kg_band": "huge", "coupling": "fair"}'
        ctx = parse_vlm_json(text)
        assert ctx.load_kg == 0.0
        assert any("unknown load_kg_band" in n for n in ctx.notes)

    def test_no_json_at_all(self):
        ctx = parse_vlm_json("I cannot answer this.")
        assert ctx.load_kg == 0.0
        assert ctx.coupling == 1
        assert "no JSON block" in ctx.notes[0]


class TestHeuristicBackend:
    def test_defaults_without_pose(self):
        h = HeuristicBackend()
        ctx = h.extract()
        assert ctx.coupling == 1               # Fair default per audit
        assert ctx.activity_static is True
        assert ctx.load_kg == 0.0

    def test_close_grip_detection(self):
        """Both wrists clustered → infer 5 kg + Good coupling."""
        pose = _upright_skeleton(upper_arm_flex_deg=80.0, elbow_flex_deg=90.0)
        kp = pose.keypoints
        # Make both wrists meet in front of the body (close grip)
        mid = 0.5 * (kp[9] + kp[10])
        kp[9] = mid + np.array([-0.02, 0.0, 0.0])    # WRIST_L
        kp[10] = mid + np.array([+0.02, 0.0, 0.0])   # WRIST_R
        ctx = HeuristicBackend().extract(pose=pose)
        assert ctx.load_kg == 5.0
        assert ctx.coupling == 0
        assert any("close-grip" in n for n in ctx.notes)

    def test_second_person_contact(self):
        """Wrist on another person's shoulder -> >20 kg assistive load; a person
        far away, or no 2D keypoints at all, leaves the load untouched; contact
        outranks the close-grip 5 kg rule."""
        from ace_pose.major_joints import SHOULDER_L, WRIST_R
        pose = _upright_skeleton()
        p2d = pose.keypoints.copy()
        other = p2d + 10.0                                   # far away
        h = HeuristicBackend()
        far = h.extract(pose=pose, worker_metadata={"primary_kpts_2d": p2d, "other_people_kpts_2d": [other]})
        assert far.load_kg == 0.0
        assert h.extract(pose=pose).load_kg == 0.0          # opt-in only
        other[SHOULDER_L] = p2d[WRIST_R]                     # touch
        near = h.extract(pose=pose, worker_metadata={"primary_kpts_2d": p2d, "other_people_kpts_2d": [other]})
        assert near.load_kg == HeuristicBackend.CONTACT_LOAD_KG and near.load_band == ">20"
        kp = pose.keypoints                                  # add a close grip too
        mid = 0.5 * (kp[9] + kp[10])
        kp[9], kp[10] = mid - 0.01, mid + 0.01
        both = h.extract(pose=pose, worker_metadata={"primary_kpts_2d": p2d, "other_people_kpts_2d": [other]})
        assert both.load_kg == HeuristicBackend.CONTACT_LOAD_KG

    def test_overhead_downgrades_coupling(self):
        """One wrist above shoulder → coupling worsens to Poor."""
        pose = _upright_skeleton(upper_arm_flex_deg=170.0)  # arm raised
        ctx = HeuristicBackend().extract(pose=pose)
        assert ctx.coupling == 2  # poor (overhead)
        assert any("overhead" in n for n in ctx.notes)


class TestOrchestrator:
    def test_default_orchestrator_uses_heuristic(self):
        pose = _upright_skeleton()
        out = ContextExtractor()(pose=pose)
        assert out.backend == "heuristic"
        assert isinstance(out, VLMContext)

    def test_kwargs_for_ergo_v2(self):
        pose = _upright_skeleton()
        ctx = ContextExtractor()(pose=pose)
        kw = ctx.to_ergo_kwargs()
        # ergo_v2.score_all accepts these specific names
        for required in ("load_kg", "coupling", "static_minute",
                         "repetitive_4_per_min", "rapid_changes"):
            assert required in kw


class TestBatchColumns:
    def test_flat_dict(self):
        ctx = VLMContext(
            backend="heuristic", load_kg=5.0, load_band="5-10",
            coupling=0, coupling_label="good",
            activity_static=True, activity_repetitive=False,
            activity_rapid=False, confidence=0.6,
            notes=["close-grip"],
        )
        cols = context_to_columns(ctx)
        assert cols["ctx_load_kg"] == 5.0
        assert cols["ctx_load_band"] == "5-10"
        assert cols["ctx_coupling"] == 0
        assert cols["ctx_activity_static"] is True
        assert cols["ctx_notes"] == "close-grip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
