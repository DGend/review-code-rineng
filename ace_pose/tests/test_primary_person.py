"""Primary-person selection must follow the labelled worker in 2D image space.

The cohort run once compared image-space label boxes with per-person 3D crop
coordinates, which picked a bystander in about a quarter of multi-person frames.
"""
import numpy as np

from ace_pose.batch_infer import COCO_TO_LABEL, _pick_primary_person, label_distance


def _person(cx, cy):
    """17 COCO body keypoints spread around (cx, cy) in image pixels."""
    rng = np.random.default_rng(int(cx))
    return np.c_[cx + rng.uniform(-60, 60, 17), cy + rng.uniform(-150, 150, 17)]


def _label_from(body, worker):
    kp = np.zeros((27, 3))
    for c, l in COCO_TO_LABEL.items():
        kp[l] = [*body[c], 2]
    x1, y1 = body.min(axis=0) - 10
    x2, y2 = body.max(axis=0) + 10
    return {"worker": worker, "bbox": [x1, y1, x2, y2], "keypoints": kp.tolist()}


def test_label_distance_is_zero_for_the_labelled_person():
    body = _person(1500, 600)
    assert label_distance(body, _label_from(body, 1)) < 1e-9


def test_picks_the_file_name_worker_not_a_more_confident_bystander():
    bystander, worker = _person(400, 500), _person(1600, 700)
    kpts2d = np.stack([np.vstack([b, np.zeros((116, 2))]) for b in (bystander, worker)])
    scores = np.array([np.full(133, 0.9), np.full(133, 0.5)])
    anns = [_label_from(bystander + 5, 2), _label_from(worker, 5)]
    idx, ann, d = _pick_primary_person(kpts2d, scores, anns, file_worker=5)
    assert (idx, ann["worker"]) == (1, 5) and d < 0.05


def test_session_code_uses_the_first_annotation():
    a, b, c = _person(300, 500), _person(1000, 600), _person(1700, 500)
    kpts2d = np.stack([np.vstack([p, np.zeros((116, 2))]) for p in (a, b, c)])
    anns = [_label_from(b, 8), _label_from(c, 7)]   # first annotation = primary worker 8
    idx, ann, d = _pick_primary_person(kpts2d, np.ones((3, 133)), anns, file_worker=103)
    assert (idx, ann["worker"]) == (1, 8) and d < 0.05


def test_without_labels_takes_the_most_confident_detection():
    kpts2d = np.stack([np.vstack([_person(x, 500), np.zeros((116, 2))]) for x in (300, 900)])
    scores = np.array([np.full(133, 0.4), np.full(133, 0.8)])
    idx, ann, d = _pick_primary_person(kpts2d, scores, None)
    assert (idx, ann) == (1, None) and np.isinf(d)
