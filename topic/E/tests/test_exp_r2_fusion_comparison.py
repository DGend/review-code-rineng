# PWW-DISCLOSURE: public
"""qwk() must equal scikit-learn's quadratic-weighted kappa; paired_ci() is deterministic."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_r2_fusion_comparison import paired_ci, qwk  # noqa: E402


def test_qwk_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 5, 300)
    sys_ = np.clip(ref + rng.integers(-1, 2, 300), 0, 4)
    lv = [0, 1, 2, 3, 4]
    assert abs(qwk(ref, sys_, lv) - sk.cohen_kappa_score(ref, sys_, labels=lv, weights="quadratic")) < 1e-12


def test_paired_ci_identical_methods_zero():
    x = np.arange(50.0)
    idx = np.random.default_rng(0).integers(0, 50, (200, 50))
    d, lo, hi = paired_ci(lambda i: x[i].mean() - x[i].mean(), idx)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)


def _synthetic(rng, n_person=3):
    """Cached detections of one image: n persons, wb 23 / body 17 / bodyfeet 23 keypoints."""
    base = rng.uniform(100, 900, (n_person, 1, 2)).astype(np.float32)
    d = {}
    for t, nk in (("wb", 23), ("b", 17), ("bf", 23)):
        d[f"im/{t}_k"] = base + rng.normal(0, 30, (n_person, nk, 2)).astype(np.float32)
        d[f"im/{t}_s"] = rng.uniform(0.1, 0.95, (n_person, nk)).astype(np.float32)
    return d


def _ann(k):  # label_distance-compatible annotation lying exactly on detection k
    from ace_pose.batch_infer import COCO_TO_LABEL
    kp = np.zeros((26, 3))
    for c, l in COCO_TO_LABEL.items():
        kp[l] = [*k[c], 2.0]
    return {"keypoints": kp, "bbox": [k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()]}


def test_audit_invariance_synthetic():
    ace = pytest.importorskip("ace_pose.bioera_v2")
    from ace_pose.batch_infer import label_distance
    import exp_r2_fusion_comparison as M
    rng = np.random.default_rng(3)
    det = _synthetic(rng)
    ann = _ann(det["im/wb_k"][1])
    # (나): annotation must not matter, for any method
    far = {**ann, "keypoints": ann["keypoints"] + np.array([800.0, 800.0, 0.0])}
    a = M.fuse_all(det, "im", ann, M.NOANN, ace, label_distance)[0]
    b = M.fuse_all(det, "im", far, M.NOANN, ace, label_distance)[0]
    assert M._same_out(a, b)
    # (가): all methods use the labelled person; the other persons must not matter
    out, _, sel = M.fuse_all(det, "im", ann, M.SAME, ace, label_distance)
    assert sel["wb"] == 1
    alt = dict(det)
    for t in ("wb", "b", "bf"):
        k = det[f"im/{t}_k"].copy()
        for q in range(len(k)):
            if q != sel[t]:
                k[q] += 77.0
        alt[f"im/{t}_k"] = k
    assert M._same_out(out, M.fuse_all(alt, "im", ann, M.SAME, ace, label_distance)[0])
    # and the annotation does matter for (가): another worker gives another skeleton
    other = M.fuse_all(det, "im", _ann(det["im/wb_k"][0]), M.SAME, ace, label_distance)[0]
    assert not M._same_out(out, other)


def test_soft_k3_equals_ensemble_soft_and_k_matters():
    """A7: soft_k_fuse(k=3) reproduces ace_pose's hard-coded soft fusion exactly (both conditions); another k changes it."""
    ace = pytest.importorskip("ace_pose.bioera_v2")
    from ace_pose.batch_infer import label_distance
    import exp_r2_fusion_comparison as M
    det = _synthetic(np.random.default_rng(5))
    ann = _ann(det["im/wb_k"][1])
    for cond in (M.SAME, M.NOANN):
        out = M.fuse_all(det, "im", ann, cond, ace, label_distance, ks=(1, 3, 10))[0]
        assert np.array_equal(out["soft_k3"][0], out["soft"][0]) and np.array_equal(out["soft_k3"][1], out["soft"][1])
        assert not np.allclose(out["soft_k10"][0], out["soft_k1"][0])


def test_cache_mismatch_classification():
    """A7: identical / same person with other numbers / another detection of the image / matches nothing."""
    import exp_r2_fusion_comparison as M
    wb = np.stack([np.full((23, 2), 100.0), np.full((23, 2), 500.0)])
    cls = lambda cached, sel, d: M.cache_mismatch_row("s", wb, cached, sel, d)["category"]  # noqa: E731
    assert cls(wb[0], 0, 0.0) == "identical"
    assert cls(wb[0] + 5, 0, 5.0) == "numeric_same_person"
    assert cls(wb[1] + 0.5, 0, 400.5) == "person_selection"
    assert cls(np.full((23, 2), 330.0), 0, 230.0) == "unmatched"      # nearest is detection 1 (170 px away), not the selected one
