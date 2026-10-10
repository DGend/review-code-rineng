# PWW-DISCLOSURE: public
"""Ergonomic joint angles from images against motion-capture ground truth (CarDA ws30).

Ground truth: the Xsens BVH of each worker (forward kinematics, carda_bvh.py), Y up, sampled at
the video rate (every second 60 Hz frame). The release aligns mp4 and BVH by hand at the first
frame; the worker's body height in the inside camera (top to bottom of the 2D keypoints) against
the head height above the toes in the BVH gives a lag of 40-60 frames (r 0.56-0.83; broad
peaks, about +-0.5 s). The lag comes from a signal no compared method uses, and is applied to
both cameras (the two ZED cameras are hardware-synchronised). Worker 003 shows no clear peak
(r <= 0.18) and is left out. Angles are measured
in 3D with the same angle functions the scorer uses (ace_pose/ergo_v2/angles.py), in a 3D body
frame. Image side: RTMW3D (the current pose model) on every frame of both cameras; the worker
is the largest person in the first frame, then the nearest pose frame to frame.

Methods compared:
  planar  the current scorer: image-plane body frame, depth dropped; it does not measure the
          wrist and assumes neutral (0 degrees), so its wrist columns are 0
  rom3d   3D body model fitted to the 2D keypoints under bone-length ratios and joint
          range-of-motion limits (ace_pose/lift3d_rom.py); angle = median over the kept
          solutions, uncertainty = their spread
  free3d  the same without range-of-motion limits (ablation)
  bal3d   rom3d plus the standing-balance prior and camera tilt (lift3d_rom.fit balance=True),
          added after the pre-registered evaluation (exploratory); run with `lift STEP bal`
  rtmw3d  RTMW3D's own 3D output, depth rescaled per frame to the standard bone proportions
  prior   no image: each angle's mean over the other workers' ground truth (leave one worker out)
Reported: mean absolute error, agreement of the OWAS/RULA/REBA angle band (BANDS), the rank
correlation of the rom3d solution spread with its error, error and coverage when frames with
a wide spread are withheld, and 95% block-bootstrap intervals (2 s blocks) of error differences.

Usage: python topic/E/src/carda_angle_eval.py infer | lift [STEP] | eval
Writes topic/E/result/carda_angles/ (poses_<clip>.npz, per_frame.csv, summary.json).
Data: datasets/private/carda/extracted/ (CC BY 4.0, not redistributed).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose.ergo_v2 import angles as A  # noqa: E402
from ace_pose.major_joints import Major25Pose, coco_wholebody_to_major25  # noqa: E402
from carda_bvh import load_bvh  # noqa: E402

WS30 = ROOT / "datasets/private/carda/extracted/ws30-svo-mp4-bvh/ws30"
OUT = ROOT / "topic/E/result/carda_angles"
BVH = {"002": "xsens_002_WS30_2023_10_04_cropped.bvh", "003": "xsens_003_WS30_2023_10_04_cropped_fixed.bvh",
       "004": "xsens_004_WS30_2023_10_04_cropped_fixed_2.bvh", "005": "xsens_005_WS30_2023_10_04_cropped_fixed.bvh"}
LAG = {"002": 40, "004": 60, "005": 45}     # video frame i <-> ground-truth frame i + LAG (30 Hz); 003 unsynced
# Major-25 joint <- BVH joint (no face points in the BVH: the head joint stands in for nose, eyes, ears)
M25_FROM_BVH = {0: "Head", 1: "Head", 2: "Head", 3: "Head", 4: "Head",
                5: "LeftShoulder", 6: "RightShoulder", 7: "LeftElbow", 8: "RightElbow",
                9: "LeftWrist", 10: "RightWrist", 11: "LeftHip", 12: "RightHip", 13: "LeftKnee",
                14: "RightKnee", 15: "LeftAnkle", 16: "RightAnkle", 17: "LeftToe", 18: "LeftToe",
                19: "LeftAnkle", 20: "RightToe", 21: "RightToe", 22: "RightAnkle", 23: "Neck", 24: "Hips"}
ANGLES = ["trunk_flexion", "trunk_lateral", "trunk_twist", "upper_arm_flexion_r", "upper_arm_flexion_l",
          "upper_arm_abduction_r", "upper_arm_abduction_l", "elbow_flexion_r", "elbow_flexion_l",
          "knee_flexion_r", "knee_flexion_l", "wrist_flexion_r", "wrist_flexion_l",
          "wrist_deviation_r", "wrist_deviation_l"]
# COCO-WholeBody hand keypoints (right hand 112-132, left 91-111): middle, index, little-finger knuckles
HAND_IDX = [112 + 9, 112 + 5, 112 + 17, 91 + 9, 91 + 5, 91 + 17]


def clips(synced_only=False):
    c = sorted((WS30 / "mp4").glob("*.mp4"))
    return [x for x in c if worker_of(x) in LAG] if synced_only else c


def worker_of(clip: Path) -> str:
    return clip.name.split("_")[1]


def frame_3d(pose: Major25Pose, world_up: np.ndarray) -> A.BodyFrame:
    """3D body frame: up = pelvis -> neck, side = left -> right shoulder (made perpendicular to up),
    fwd perpendicular to both, signed toward the pelvis's facing direction (world up x hip line).
    The toes were used first and failed in running, where the swing foot points backwards.
    `anterior` is the horizontal facing direction: the trunk-flexion sign needs it, since fwd
    is perpendicular to the trunk itself."""
    kp = pose.keypoints
    up = A._norm(kp[A.NECK] - kp[A.PELVIS])
    side = A._norm(A._project_perpendicular(kp[A.SHOULDER_R] - kp[A.SHOULDER_L], up))
    fwd = A._norm(np.cross(side, up))
    facing = np.cross(A._norm(world_up), kp[A.HIP_R] - kp[A.HIP_L])
    if np.dot(fwd, facing) < 0:
        fwd = -fwd
    anterior = A._norm(A._project_perpendicular(fwd, A._norm(world_up)))
    return A.BodyFrame(up=up, side=side, fwd=fwd, pelvis=kp[A.PELVIS], planar=False, anterior=anterior)


def measure(pose: Major25Pose, frame: A.BodyFrame, world_up: np.ndarray, hands=None) -> dict:
    """Angles from a Major-25 pose; wrist angles from `hands` (2, 3, 3: right, left; middle,
    index, little knuckle), 0 (neutral) without hands."""
    from ace_pose.lift3d_rom import wrist_angles
    kp, wr = pose.keypoints, {}
    for k, (el, w, s) in enumerate(((A.ELBOW_R, A.WRIST_R, "r"), (A.ELBOW_L, A.WRIST_L, "l"))):
        f, d = wrist_angles(kp[el], kp[w], *hands[k], s) if hands is not None else (0.0, 0.0)
        wr[f"wrist_flexion_{s}"], wr[f"wrist_deviation_{s}"] = f, d
    return {**wr,"trunk_flexion": A.trunk_flexion_deg(pose, frame, world_up=world_up),
            "trunk_lateral": A.trunk_lateral_deg(pose, frame, world_up=world_up),
            "trunk_twist": A.trunk_twist_deg(pose, frame),
            **{f"upper_arm_flexion_{s[0]}": A.upper_arm_flexion_deg(pose, frame, side=s) for s in ("right", "left")},
            **{f"upper_arm_abduction_{s[0]}": A.upper_arm_abduction_deg(pose, frame, side=s) for s in ("right", "left")},
            **{f"elbow_flexion_{s[0]}": A.elbow_flexion_deg(pose, side=s) for s in ("right", "left")},
            **{f"knee_flexion_{s[0]}": A.knee_flexion_deg(pose, side=s) for s in ("right", "left")}}


def virtual_hands(wrist_pos, wrist_rot, rest_dir_r, rest_dir_l, thumb):
    """Knuckles (2, 3, 3) from the global wrist rotations of a skeleton whose rest pose has the
    hands along rest_dir, thumbs along `thumb` and palms down (checked: hanging hands then face
    the thigh), with the model's hand geometry (lift3d_rom.HAND)."""
    from ace_pose.lift3d_rom import HAND
    out = np.zeros((2, 3, 3))
    for k, (pos, R, d) in enumerate(zip(wrist_pos, wrist_rot, (rest_dir_r, rest_dir_l))):
        local = -HAND[:, 1:2] * np.asarray(d, float) + HAND[:, 2:3] * np.asarray(thumb, float)
        out[k] = pos + local @ R.T
    return out


def ground_truth(worker: str):
    names, pos, fps, rot = load_bvh(WS30 / "xsens" / BVH[worker], with_rotations=True)
    pos, rot = pos[::fps // 30], rot[::fps // 30]
    wi = [names.index("RightWrist"), names.index("LeftWrist")]
    idx = [names.index(M25_FROM_BVH[i]) for i in range(25)]
    up = np.array([0.0, 1.0, 0.0])
    out = []
    for p, r in zip(pos, rot):
        pose = Major25Pose(keypoints=p[idx], confidence=np.ones(25), source=np.zeros(25, np.int8))
        hands = virtual_hands(p[wi], r[wi], (-1, 0, 0), (1, 0, 0), (0, 0, 1))
        out.append(measure(pose, frame_3d(pose, up), up, hands))
    return out


def infer():
    import cv2
    import rtmlib
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    import os
    OUT.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        model = rtmlib.Wholebody3d(mode="balanced", backend="onnxruntime",
                                   device=os.environ.get("BIOERA_DEVICE", "cuda"))
    for clip in clips():
        out = OUT / f"poses_{clip.stem}.npz"
        if out.exists():
            continue
        cap, prev, store = cv2.VideoCapture(str(clip)), None, {"k3": [], "k2": [], "s": [], "found": []}
        while True:
            ok, img = cap.read()
            if not ok:
                break
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                k3, sc, _simcc, k2 = model(img)
            size = [np.prod(np.ptp(p[:17], axis=0)) for p in k2]
            keep = [i for i, a in enumerate(size) if a > 400]          # drops empty placeholder poses
            j = None
            if keep and prev is None:
                j = max(keep, key=lambda i: size[i])
            elif keep:
                d = {i: np.median(np.linalg.norm(k2[i][:17] - prev, axis=1)) for i in keep}
                i = min(d, key=d.get)
                j = i if d[i] <= 0.5 * np.hypot(*np.ptp(prev, axis=0)) else None
            prev = k2[j][:17] if j is not None else None
            store["found"].append(j is not None)
            for key, arr, shape in (("k3", k3, (133, 3)), ("k2", k2, (133, 2)), ("s", sc, (133,))):
                store[key].append(arr[j] if j is not None else np.full(shape, np.nan))
        np.savez_compressed(out, **{k: np.asarray(v, np.float32) for k, v in store.items()})
        print(f"{clip.stem}: {len(store['found'])} frames, worker found {np.mean(store['found']):.3f}", flush=True)


# score bands used by RULA/REBA for each angle (upper bounds, degrees; see ace_pose/standards)
def _band_trunk(x):        # REBA trunk: upright, 0-20, 20-60 (or extension > 20), > 60
    return 1 if abs(x) <= 5 else 2 if abs(x) <= 20 else 4 if x > 60 else 3


def _band_upper_arm(x):    # RULA/REBA upper arm: -20..20, ext > 20 or 20-45, 45-90, > 90
    return 1 if -20 <= x <= 20 else 2 if x < -20 or x <= 45 else 3 if x <= 90 else 4


def _band_knee(x):         # REBA legs adjustment: < 30, 30-60, > 60
    return 0 if x < 30 else 1 if x <= 60 else 2


def _band_wrist(x):        # REBA wrist: 0-15 flexion or extension, > 15
    return 1 if abs(x) <= 15 else 2


def _band_elbow(x):        # RULA/REBA lower arm: 60-100 neutral
    return 1 if 60 <= x <= 100 else 2


BANDS = {"trunk_flexion": _band_trunk, "upper_arm_flexion_r": _band_upper_arm, "upper_arm_flexion_l": _band_upper_arm,
         "knee_flexion_r": _band_knee, "knee_flexion_l": _band_knee,
         "elbow_flexion_r": _band_elbow, "elbow_flexion_l": _band_elbow,
         "wrist_flexion_r": _band_wrist, "wrist_flexion_l": _band_wrist}


def hands_2d(k133, s133):
    """(6, 2) knuckles and (6,) scores from COCO-WholeBody keypoints, in lift3d_rom order."""
    return np.asarray(k133)[HAND_IDX, :2], np.asarray(s133)[HAND_IDX]


def rtmw3d_angles(k3, sc):
    """Angles from RTMW3D's 3D output; its depth is in other units than x, y (pixels), so it is
    rescaled per frame to the factor that best matches the standard bone proportions."""
    from ace_pose import lift3d_rom as L
    m = coco_wholebody_to_major25(k3, sc)
    kp = m.keypoints.copy()
    bones = [(A.SHOULDER_R, A.ELBOW_R, "upper_arm"), (A.SHOULDER_L, A.ELBOW_L, "upper_arm"),
             (A.ELBOW_R, A.WRIST_R, "forearm"), (A.ELBOW_L, A.WRIST_L, "forearm"),
             (A.HIP_R, A.KNEE_R, "thigh"), (A.HIP_L, A.KNEE_L, "thigh"),
             (A.KNEE_R, A.ANKLE_R, "shank"), (A.KNEE_L, A.ANKLE_L, "shank"), (A.PELVIS, A.NECK, "trunk")]
    best = None
    for c in np.geomspace(1, 3000, 60):
        q = kp * [1, 1, c]
        lens = np.array([np.linalg.norm(q[a] - q[b]) for a, b, _ in bones])
        ratio = lens / lens[-1] - np.array([L.SEGMENTS[n] / L.SEGMENTS["trunk"] for *_, n in bones])
        cost = np.sum(ratio ** 2)
        if best is None or cost < best[0]:
            best = (cost, q, c)
    pose = Major25Pose(keypoints=best[1], confidence=m.confidence, source=m.source)
    hands = (np.asarray(k3, float)[HAND_IDX] * [1, 1, best[2]]).reshape(2, 3, 3)
    up = np.array([0.0, -1.0, 0.0])
    return measure(pose, frame_3d(pose, up), up, hands)


PLAIN = (("rom3d", True, False), ("free3d", False, False))
BALANCED = (("bal3d", True, True),)


def _lift_one(args, variants=PLAIN):
    from ace_pose import lift3d_rom as L
    kp, conf, h2, hc = args
    up, out = np.array([0.0, 1.0, 0.0]), {}
    for name, rom, bal in variants:
        try:
            fit = L.fit(kp, conf, rom=rom, hands2d=h2, hands_conf=hc, balance=bal)
        except np.linalg.LinAlgError:              # counted and reported, the frame is left out
            fit = L.FitResult([], [], [], [])
        if not fit.body_joints:
            out.update({f"{name}_{k}": np.nan for k in ANGLES})
            out[f"{name}_solutions"] = 0
            continue
        per = []
        for j, hd in zip(fit.body_joints, fit.hand_joints):
            pose = Major25Pose(keypoints=j, confidence=np.ones(25), source=np.zeros(25, np.int8))
            per.append(measure(pose, frame_3d(pose, up), up, hd))
        for k in ANGLES:
            v = [d[k] for d in per]
            out[f"{name}_{k}"] = float(np.median(v))
            out[f"{name}_{k}_spread"] = float(np.ptp(v))
        out[f"{name}_solutions"] = len(per)
    return out


def _lift_bal(args):
    return _lift_one(args, BALANCED)


def lift(step=3, which="plain"):
    """Fit the 3D body model on every `step`-th frame where the worker was found
    (which="bal": the balance variant only, written to liftbal_<clip>.csv)."""
    import pandas as pd
    from multiprocessing import Pool
    fn, prefix = (_lift_one, "lift") if which == "plain" else (_lift_bal, "liftbal")
    for clip in clips(synced_only=True):
        out = OUT / f"{prefix}_{clip.stem}.csv"
        if out.exists():
            continue
        z = np.load(OUT / f"poses_{clip.stem}.npz")
        frames = [i for i in range(0, len(z["found"]), step) if z["found"][i]]
        jobs = []
        for i in frames:
            m25 = coco_wholebody_to_major25(np.c_[z["k2"][i], np.zeros(133)], z["s"][i])
            jobs.append((m25.keypoints[:, :2], m25.confidence, *hands_2d(z["k2"][i], z["s"][i])))
        with Pool(14) as pool:
            res = pool.map(fn, jobs, chunksize=4)
        pd.DataFrame([{"frame": i, **r} for i, r in zip(frames, res)]).to_csv(out, index=False, float_format="%.2f")
        failed = sum(next(v for k, v in r.items() if k.endswith("_solutions")) == 0 for r in res)
        print(f"{clip.stem}: {len(frames)} frames lifted, {failed} without a 3D fit", flush=True)


def evaluate():
    import pandas as pd
    rows = []
    for clip in clips(synced_only=True):
        w, view = worker_of(clip), ("in" if "_in_" in clip.name else "out")
        gt = ground_truth(w)[LAG[w]:]
        z = np.load(OUT / f"poses_{clip.stem}.npz")
        img_up = np.array([0.0, -1.0, 0.0])
        for i in range(min(len(gt), len(z["found"]))):
            if not z["found"][i]:
                continue
            k2 = np.c_[z["k2"][i], np.zeros(133)]
            m25 = coco_wholebody_to_major25(k2, z["s"][i])
            planar = measure(m25, A.build_body_frame(m25), img_up)
            sw = np.linalg.norm(m25.keypoints[A.SHOULDER_R, :2] - m25.keypoints[A.SHOULDER_L, :2])
            leg = np.linalg.norm(m25.keypoints[A.PELVIS, :2] - 0.5 * (m25.keypoints[A.ANKLE_L, :2] + m25.keypoints[A.ANKLE_R, :2]))
            row = {"clip": clip.stem, "worker": w, "view": view, "frame": i, "t": i / 30,
                   "side_view": bool(sw / max(leg, 1e-6) <= 0.24)}
            row.update({f"gt_{k}": v for k, v in gt[i].items()})
            row.update({f"planar_{k}": v for k, v in planar.items()})
            row.update({f"rtmw3d_{k}": v for k, v in rtmw3d_angles(z["k3"][i], z["s"][i]).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    lifted = [pd.read_csv(f).assign(clip=f.stem[5:]) for f in sorted(OUT.glob("lift_*.csv"))]
    if lifted:                         # the 3D methods are evaluated on the lifted frames only
        df = df.merge(pd.concat(lifted), on=["clip", "frame"], how="inner")
    balanced = [pd.read_csv(f).assign(clip=f.stem[8:]) for f in sorted(OUT.glob("liftbal_*.csv"))]
    if balanced:
        df = df.merge(pd.concat(balanced), on=["clip", "frame"], how="inner")
    for k in ANGLES:                      # leave-one-worker-out mean of the ground truth
        df[f"prior_{k}"] = df.worker.map({w: df.loc[df.worker != w, f"gt_{k}"].mean() for w in df.worker.unique()})
    methods = ["planar", "rtmw3d", "prior"] + [m for m in ("rom3d", "free3d", "bal3d") if f"{m}_trunk_flexion" in df]
    df.to_csv(OUT / "per_frame.csv", index=False, float_format="%.2f")
    report(df, methods, OUT, subsets=(("first_40s", df.t < 40), ("side_view", df.side_view),
                                      ("not_side_view", ~df.side_view), ("inside_camera", df.view == "in"),
                                      ("outside_camera", df.view == "out")))


def report(df, methods, out_dir, subsets=(), group="worker"):
    """Error, band agreement, kappa, correlation, rom3d uncertainty and bootstrap intervals;
    writes out_dir/summary.json. `df` has gt_<angle> and <method>_<angle> columns, plus clip and
    frame (for 2 s bootstrap blocks)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score
    df = df.dropna(subset=[f"{m}_trunk_flexion" for m in methods]).copy()     # same frames for every method
    err = lambda d, m, k: np.abs(d[f"{m}_{k}"] - d[f"gt_{k}"])
    band = lambda d, m, k: np.mean([BANDS[k](a) == BANDS[k](b) for a, b in zip(d[f"{m}_{k}"], d[f"gt_{k}"])])
    kappa = lambda d, m, k: cohen_kappa_score(d[f"gt_{k}"].map(BANDS[k]), d[f"{m}_{k}"].map(BANDS[k]))
    corr = lambda d, m, k: np.corrcoef(d[f"gt_{k}"], d[f"{m}_{k}"])[0, 1] if d[f"{m}_{k}"].std() > 0 else 0.0
    summ = {}
    for subset, d in (("all", df),) + tuple((n, df[mask.reindex(df.index, fill_value=False)]) for n, mask in subsets):
        summ[subset] = {"frames": int(len(d)), group + "s": sorted(map(str, d[group].unique())),
                        "mae": {m: {k: float(err(d, m, k).mean()) for k in ANGLES} for m in methods},
                        "band_agreement": {m: {k: float(band(d, m, k)) for k in BANDS} for m in methods},
                        "band_kappa": {m: {k: float(kappa(d, m, k)) for k in BANDS} for m in methods},
                        "pearson_r": {m: {k: float(corr(d, m, k)) for k in BANDS} for m in methods}}
    if "rom3d" in methods:
        unc = {}
        for k in BANDS:
            e, sp = err(df, "rom3d", k), df[f"rom3d_{k}_spread"]
            rho, pv = spearmanr(sp, e)
            keep = sp <= 15.0
            unc[k] = {"spearman_rho": float(rho), "p": float(pv), "coverage_spread_le_15": float(keep.mean()),
                      "mae_kept": float(e[keep].mean()), "mae_withheld": float(e[~keep].mean()),
                      "band_agreement_kept": float(band(df[keep], "rom3d", k)),
                      "band_kappa_kept": float(kappa(df[keep], "rom3d", k))}
        summ["rom3d_uncertainty"] = unc
        # 95% block bootstrap (2 s blocks within each clip) of mean-error differences
        df["block"] = df["clip"].astype(str) + "_" + (df["frame"] // 60).astype(str)
        blocks = df.groupby("block").indices
        names, rng, diffs = list(blocks), np.random.default_rng(20261009), {}
        for a, b in [("rom3d", m) for m in methods if m != "rom3d"]:
            dk = {k: (err(df, a, k) - err(df, b, k)).to_numpy() for k in ANGLES}
            boot = {k: [] for k in ANGLES}
            for _ in range(2000):
                idx = np.concatenate([blocks[n] for n in rng.choice(names, len(names))])
                for k in ANGLES:
                    boot[k].append(dk[k][idx].mean())
            diffs[f"{a}_minus_{b}"] = {k: [float(dk[k].mean()), *map(float, np.percentile(boot[k], [2.5, 97.5]))]
                                       for k in ANGLES}
        summ["mae_difference_ci95"] = diffs
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    for s_, v in summ.items():
        if s_ in ("rom3d_uncertainty", "mae_difference_ci95"):
            continue
        print(s_, v["frames"], v[group + "s"])
        print(f"  {'':22s} MAE: " + " ".join(f"{m:>7s}" for m in methods) + "  | kappa: " + " ".join(f"{m:>7s}" for m in methods)
              + "  | r: " + " ".join(f"{m:>7s}" for m in methods))
        for k in ANGLES:
            line = f"  {k:22s}      " + " ".join(f"{v['mae'][m][k]:7.1f}" for m in methods)
            if k in BANDS:
                line += "  |        " + " ".join(f"{v['band_kappa'][m][k]:7.2f}" for m in methods)
                line += "  |    " + " ".join(f"{v['pearson_r'][m][k]:7.2f}" for m in methods)
            print(line)
    if "rom3d_uncertainty" in summ:
        print("rom3d uncertainty (spread vs error):")
        for k, u in summ["rom3d_uncertainty"].items():
            print(f"  {k:22s} rho {u['spearman_rho']:.2f}  kept {u['coverage_spread_le_15']:.0%}: MAE {u['mae_kept']:.1f} "
                  f"(withheld {u['mae_withheld']:.1f}), kappa {u['band_kappa_kept']:.2f}")
        for comp, v in summ["mae_difference_ci95"].items():
            print(comp, {k: [round(x, 1) for x in val] for k, val in v.items() if k in BANDS})
    return summ


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "lift":
        lift(int(sys.argv[2]) if len(sys.argv) > 2 else 3, sys.argv[3] if len(sys.argv) > 3 else "plain")
    else:
        {"infer": infer, "eval": evaluate}.get(cmd, lambda: sys.exit(__doc__))()
