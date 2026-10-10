# PWW-DISCLOSURE: public
"""Ergonomic joint angles from images against 3DPW ground truth (exact frame sync).

Same methods and metrics as carda_angle_eval.py (planar, rtmw3d, rom3d, free3d, prior). Ground
truth: the SMPL joints of each actor (world coordinates, y up) on the 17 test sequences and
300-frame windows used in video_pose_eval.py; wrist angles from the global SMPL wrist rotation
(forward kinematics of the pose parameters; checked against the released joints, forearm
direction cosine 0.999) with the model's hand geometry. Image side: RTMW3D (the current model)
re-run on those frames keeping all 133 keypoints (hands included); each actor is matched to
the nearest detection in 2D (as in video_pose_eval.py). The prior is the mean over the other
sequences (leave one sequence out).

Usage: python topic/E/src/threedpw_angle_eval.py [--split=validation] infer | lift [STEP] | eval
(the validation split needs `video_pose_eval.py frames3dpw validation` first)
Writes topic/E/result/threedpw_angles/. Data: datasets/private/3dpw/ (research licence, not redistributed).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carda_angle_eval import (ANGLES, ROOT, _lift_one, frame_3d, hands_2d, measure, report,  # noqa: E402
                              rtmw3d_angles, virtual_hands)
from ace_pose.ergo_v2 import angles as A  # noqa: E402
from ace_pose.major_joints import Major25Pose, coco_wholebody_to_major25  # noqa: E402
from video_pose_eval import OUT as VOUT, match  # noqa: E402

# --split validation (first argument) runs the held-out validation split into OUT/validation
SPLIT = "validation" if "--split=validation" in sys.argv else "test"
OUT = ROOT / "topic/E/result/threedpw_angles" / ("validation" if SPLIT == "validation" else "")
SEQ = ROOT / f"datasets/private/3dpw/extracted/sequenceFiles/{SPLIT}"
PREFIX = "3dpw" if SPLIT == "test" else "3dpwval"
# Major-25 joint <- SMPL joint (the head joint stands in for the face points; foot joints for toes)
M25_FROM_SMPL = [15, 15, 15, 15, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 10, 10, 7, 11, 11, 8, 12, 0]
HYBRID_3D = {"trunk_flexion", "trunk_lateral", "trunk_twist", "upper_arm_abduction_r", "upper_arm_abduction_l"}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def infer():
    """RTMW3D on the 3DPW frames, all people, all 133 keypoints (2D and 3D)."""
    import os
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    import cv2
    import rtmlib
    OUT.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        m = rtmlib.Wholebody3d(mode="balanced", backend="onnxruntime", device=os.environ.get("BIOERA_DEVICE", "cuda"))
    for clip in sorted((VOUT / "frames").glob(f"{PREFIX}_*")):
        out = OUT / f"pred_{clip.name}.npz"
        if out.exists():
            continue
        store = {}
        for f in sorted(clip.glob("*.jpg")):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                k3, sc, _simcc, k2 = m(cv2.imread(str(f)))
            store[f"{f.stem}/k2"], store[f"{f.stem}/k3"], store[f"{f.stem}/s"] = (
                np.asarray(k2, np.float32), np.asarray(k3, np.float32), np.asarray(sc, np.float32))
        np.savez_compressed(out, **store)
        print(f"{clip.name}: {len(store) // 3} frames", flush=True)


def detections():
    """(sequence, frame, actor) -> matched RTMW3D detection (Major-25 pose, 133 2D keypoints,
    133 3D keypoints, scores), for every annotated actor."""
    for clip in sorted((VOUT / "frames").glob(f"{PREFIX}_*")):
        gt, pr = np.load(clip / "gt.npz"), np.load(OUT / f"pred_{clip.name}.npz")
        for key in sorted(k for k in gt.files if k.endswith("/uv")):
            i, pid = map(int, key.split("/")[:2])
            uv, vis = gt[key], gt[f"{i}/{pid}/vis"]
            k, k3, s = pr[f"{i:08d}/k2"], pr[f"{i:08d}/k3"], pr[f"{i:08d}/s"]
            j, dist = match(k, uv, vis)
            if j is None or dist / np.hypot(*np.ptp(uv[vis], axis=0)) >= 0.1:
                continue
            yield clip.name[len(PREFIX) + 1:], i, pid, (coco_wholebody_to_major25(np.c_[k[j], np.zeros(133)], s[j]), k[j], k3[j], s[j])


def ground_truth(seq, i, pid, cache={}):
    from scipy.spatial.transform import Rotation
    if seq not in cache:
        d = pickle.load(open(SEQ / f"{seq}.pkl", "rb"), encoding="latin1")
        cache[seq] = (d["jointPositions"], d["poses"])
    pos = cache[seq][0][pid][i].reshape(24, 3)
    local = Rotation.from_rotvec(np.asarray(cache[seq][1][pid][i]).reshape(24, 3)).as_matrix()
    G = np.zeros((24, 3, 3))
    for j, par in enumerate(SMPL_PARENTS):
        G[j] = local[j] if par < 0 else G[par] @ local[j]
    pose = Major25Pose(keypoints=pos[M25_FROM_SMPL], confidence=np.ones(25), source=np.zeros(25, np.int8))
    hands = virtual_hands(pos[[21, 20]], G[[21, 20]], (-1, 0, 0), (1, 0, 0), (0, 0, 1))   # SMPL rest: T-pose, palms down
    up = np.array([0.0, 1.0, 0.0])
    return measure(pose, frame_3d(pose, up), up, hands)


def lift(step=3):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "lift.csv"
    if out.exists():
        sys.exit(f"{out} exists; delete it to re-run")
    keys, jobs = [], []
    for seq, i, pid, (m25, k2, _k3, s) in detections():
        if i % step == 0:
            keys.append((seq, i, pid))
            jobs.append((m25.keypoints[:, :2], m25.confidence, *hands_2d(k2, s)))
    with Pool(14) as pool:
        res = pool.map(_lift_one, jobs, chunksize=4)
    pd.DataFrame([{"clip": s, "frame": i, "actor": p, **r} for (s, i, p), r in zip(keys, res)]).to_csv(
        out, index=False, float_format="%.2f")
    print(f"{len(keys)} person-frames lifted, {sum(r['rom3d_solutions'] == 0 for r in res)} without a 3D fit")


def evaluate():
    img_up = np.array([0.0, -1.0, 0.0])
    rows = []
    for seq, i, pid, (m25, _k2, k3, s) in detections():
        row = {"clip": seq, "frame": i, "actor": pid}
        row.update({f"gt_{k}": v for k, v in ground_truth(seq, i, pid).items()})
        row.update({f"planar_{k}": v for k, v in measure(m25, A.build_body_frame(m25), img_up).items()})
        row.update({f"rtmw3d_{k}": v for k, v in rtmw3d_angles(k3, s).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if (OUT / "lift.csv").exists():
        df = df.merge(pd.read_csv(OUT / "lift.csv"), on=["clip", "frame", "actor"], how="inner")
    for k in ANGLES:
        df[f"prior_{k}"] = df["clip"].map({s: df.loc[df["clip"] != s, f"gt_{k}"].mean() for s in df["clip"].unique()})
    if "rom3d_trunk_flexion" in df:      # pre-registered hybrid (strategy-log S2-b): 3D for these, 2D elsewhere
        for k in ANGLES:
            df[f"hybrid_{k}"] = df[f"rom3d_{k}"] if k in HYBRID_3D else df[f"planar_{k}"]
    methods = ["planar", "rtmw3d", "prior"] + [m for m in ("rom3d", "free3d", "hybrid") if f"{m}_trunk_flexion" in df]
    df.to_csv(OUT / "per_frame.csv", index=False, float_format="%.2f")
    report(df, methods, OUT, group="clip")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--split")]
    cmd = args[0] if args else ""
    if cmd == "lift":
        lift(int(args[1]) if len(args) > 1 else 3)
    elif cmd == "infer":
        infer()
    elif cmd == "eval":
        evaluate()
    else:
        sys.exit(__doc__)
