# PWW-DISCLOSURE: public
"""Video comparison of pose models on CMU Panoptic (with ground truth) and CarDA (factory, no ground truth).

3DPW (outdoor everyday video from a hand-held phone, test split, 24 sequences): the SMPL joints
projected with the per-frame camera pose are the ground truth for shoulders to ankles (hips and head
left out, see threedpw_frames); same metrics as Panoptic, on WINDOW frames per sequence.
Panoptic: the 3D ground truth (hdPose3d_stage1_coco19) is projected into the HD camera 00_00 with
the released calibration; for each sequence a window of WINDOW consecutive frames with at least
two people in view is used. Metrics per model: PCK@0.05/0.10 of the person box diagonal, the share
of ground-truth people detected, and temporal error: joint velocity error and jitter (mean
second difference) against the ground-truth trajectories.
CarDA: consecutive frames of the released mp4 clips; per model the jitter of the most central
worker (tracked from the largest person by nearest pose), per camera: share of frames found, median jitter
(no ground truth exists: the released Xsens BVH is 3D motion capture, not image keypoints).

Usage (in an environment with rtmlib/onnxruntime and, for DETRPose, PyTorch):
  python topic/E/src/video_pose_eval.py frames            # extract Panoptic and CarDA frames, Panoptic 2D ground truth
  python topic/E/src/video_pose_eval.py frames3dpw [validation]   # 3DPW test (or validation) frames and 2D ground truth
  python topic/E/src/video_pose_eval.py infer MODEL       # MODEL as in compare_pose_models_infer.py, or rtmw3d, detrpose_l, detrpose_x
  python topic/E/src/video_pose_eval.py eval
Data: datasets/private/cmu_panoptic/<seq>/, datasets/private/carda/, datasets/private/3dpw/ (not redistributed).
Writes summary.json and summary_by_model.csv (one row per model).
Writes topic/E/result/video_pose_comparison/.
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = ROOT / "datasets/private"
OUT = ROOT / "topic/E/result/video_pose_comparison"
SEQS = ("160422_haggling1", "160906_pizza1", "170221_haggling_b1")
CAM, WINDOW = "00_00", 300
# Panoptic COCO-19 index for each COCO-17 joint
P19_TO_COCO17 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
# SMPL joint index for COCO-17 joints 5-16 (shoulders to ankles); the head joints are not in SMPL
SMPL_TO_COCO17 = [None] * 5 + [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


def project(X, cam):
    """World points (N, 3, cm) to pixel coordinates in an HD camera, with lens distortion."""
    R, t, K, d = (np.array(cam[k]) for k in ("R", "t", "K", "distCoef"))
    x = R @ X.T + t.reshape(3, 1)
    z = x[2]
    a, b = x[0] / z, x[1] / z
    r2 = a * a + b * b
    radial = 1 + d[0] * r2 + d[1] * r2 ** 2 + d[4] * r2 ** 3
    ad = a * radial + 2 * d[2] * a * b + d[3] * (r2 + 2 * a * a)
    bd = b * radial + d[2] * (r2 + 2 * b * b) + 2 * d[3] * a * b
    return np.stack([K[0, 0] * ad + K[0, 2], K[1, 1] * bd + K[1, 2]], axis=1), z


def panoptic_frames():
    for seq in SEQS:
        d = DATA / "cmu_panoptic" / seq
        cam = next(c for c in json.load(open(d / f"calibration_{seq}.json"))["cameras"] if c["name"] == CAM)
        files = sorted(glob.glob(str(d / "hdPose3d_stage1_coco19" / "body3DScene_*.json")))
        gt = {}
        for f in files:
            idx = int(Path(f).stem.split("_")[-1])
            try:
                bodies = json.load(open(f)).get("bodies", [])
            except json.JSONDecodeError:      # a few released files are empty; the frame is left out
                continue
            people = []
            for b in bodies:
                j = np.array(b["joints19"]).reshape(-1, 4)[P19_TO_COCO17]
                uv, z = project(j[:, :3], cam)
                vis = (j[:, 3] > 0.2) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
                if vis.sum() >= 8:
                    people.append((b["id"], uv, vis))
            gt[idx] = people
        idxs = sorted(gt)
        # first window of WINDOW consecutive frames with >= 2 people in view throughout
        start = next(i for k, i in enumerate(idxs[:-WINDOW]) if idxs[k + WINDOW - 1] - i == WINDOW - 1
                     and all(len(gt[i + w]) >= 2 for w in range(WINDOW)))
        frames = list(range(start, start + WINDOW))
        cap = cv2.VideoCapture(str(d / "videos" / f"hd_{CAM}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        out = OUT / "frames" / f"panoptic_{seq}"
        out.mkdir(parents=True, exist_ok=True)
        for i in frames:
            ok, img = cap.read()
            if not ok:
                sys.exit(f"{seq}: video ended at frame {i}")
            cv2.imwrite(str(out / f"{i:08d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savez_compressed(out / "gt.npz", **{f"{i}/{pid}/uv": uv for i in frames for pid, uv, _ in gt[i]},
                            **{f"{i}/{pid}/vis": vis for i in frames for pid, _, vis in gt[i]})
        print(f"{seq}: frames {frames[0]}-{frames[-1]}, people per frame {np.mean([len(gt[i]) for i in frames]):.1f}")


def threedpw_frames(split="test"):
    """3DPW test split (folders 3dpw_<seq>; the validation split goes to 3dpwval_<seq> and is kept
    out of this script's video comparison): the SMPL 3D joints projected with the per-frame camera pose are the ground
    truth (the released poses2D are OpenPose detections, not ground truth). Hips are left out: the
    SMPL hip joint lies inside the pelvis, not at the COCO hip point."""
    import pickle
    import shutil
    root = DATA / "3dpw" / "extracted"
    prefix = "3dpw" if split == "test" else "3dpwval"
    for f in sorted((root / "sequenceFiles" / split).glob("*.pkl")):
        d = pickle.load(open(f, "rb"), encoding="latin1")
        seq, K, n = d["sequence"], np.asarray(d["cam_intrinsics"]), len(d["cam_poses"])
        valid = np.asarray(d["campose_valid"]).astype(bool).all(axis=0)
        # the window of WINDOW consecutive valid frames closest to the middle of the sequence
        ok = [s for s in range(n - WINDOW + 1) if valid[s:s + WINDOW].all()]
        if not ok:
            print(f"{seq}: no {WINDOW}-frame window with a valid camera pose, skipped")
            continue
        start = min(ok, key=lambda s: abs(s + WINDOW // 2 - n // 2))
        frames = list(range(start, start + WINDOW))
        h, w = cv2.imread(str(root / "imageFiles" / seq / f"image_{start:05d}.jpg")).shape[:2]
        out = OUT / "frames" / f"{prefix}_{seq}"
        out.mkdir(parents=True, exist_ok=True)
        gt = {}
        for i in frames:
            shutil.copyfile(root / "imageFiles" / seq / f"image_{i:05d}.jpg", out / f"{i:08d}.jpg")
            for pid, J in enumerate(d["jointPositions"]):
                X = np.c_[J[i].reshape(24, 3)[SMPL_TO_COCO17[5:]], np.ones(12)]
                x = (d["cam_poses"][i] @ X.T)[:3]
                uv = np.zeros((17, 2))
                uv[5:] = (K @ (x / x[2]))[:2].T
                vis = np.zeros(17, bool)
                vis[5:] = (x[2] > 0) & (uv[5:, 0] >= 0) & (uv[5:, 0] < w) & (uv[5:, 1] >= 0) & (uv[5:, 1] < h)
                vis[[11, 12]] = False
                if vis.sum() >= 6:
                    gt[f"{i}/{pid}/uv"], gt[f"{i}/{pid}/vis"] = uv, vis
        np.savez_compressed(out / "gt.npz", **gt)
        print(f"{seq}: frames {frames[0]}-{frames[-1]}, actors {len(d['jointPositions'])}, {w}x{h}")


def carda_frames():
    clips = sorted(glob.glob(str(DATA / "carda" / "extracted" / "**" / "*.mp4"), recursive=True))
    for c in clips:
        cap = cv2.VideoCapture(c)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2 - WINDOW // 2))
        out = OUT / "frames" / f"carda_{Path(c).stem.replace(' ', '_')}"
        out.mkdir(parents=True, exist_ok=True)
        for i in range(WINDOW):
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{i:08d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"carda {Path(c).name}: {total} frames, kept {i + 1} around the middle")


def build(name, device):
    if name == "rtmw3d":
        import rtmlib
        m = rtmlib.Wholebody3d(mode="balanced", backend="onnxruntime", device=device)

        def run(img):
            _k3, sc, _simcc, k2 = m(img)
            return k2[:, :23], sc[:, :23]
        return run
    if name.startswith("detrpose"):
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from compare_pose_models_detrpose import KEEP, load
        dev = "cuda:0" if device == "cuda" else device
        mod, tf = load(name[-1], dev), T.Compose([T.Resize((640, 640)), T.ToTensor()])

        @torch.no_grad()
        def run(img):
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            s, _l, k = mod(tf(im).unsqueeze(0).to(dev), torch.tensor([list(im.size)]).to(dev))
            s, k = s[0].cpu().numpy(), k[0].cpu().numpy()[..., :2]
            keep = s > KEEP
            return k[keep], np.repeat(s[keep, None], k.shape[1], axis=1)
        return run
    os.environ["BIOERA_DEVICE"] = device
    from compare_pose_models_infer import build as b
    m, n = b(name)

    def run(img):
        k, s = m(img)
        k = np.asarray(k, float).reshape(-1, np.shape(k)[-2] if np.ndim(k) == 3 else 0, 2)
        return k[:, :n], np.asarray(s, float).reshape(len(k), -1)[:, :n]
    return run


def infer(name, device=os.environ.get("BIOERA_DEVICE", "cuda")):
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        run = build(name, device)
    for clip in sorted((OUT / "frames").iterdir()):
        out = OUT / "pred" / name / f"{clip.name}.npz"
        if out.exists():
            continue
        store = {}
        for f in sorted(clip.glob("*.jpg")):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                k, s = run(cv2.imread(str(f)))
            store[f"{f.stem}/k"], store[f"{f.stem}/s"] = np.asarray(k, np.float32), np.asarray(s, np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **store)
        print(f"{name} {clip.name}: {len(store) // 2} frames")


def match(pred, gt_uv, gt_vis):
    """Index of the predicted person closest to a ground-truth person (median distance over visible joints)."""
    if len(pred) == 0:
        return None, np.inf
    d = [np.median(np.linalg.norm(p[:17][gt_vis] - gt_uv[gt_vis], axis=1)) for p in pred]
    i = int(np.argmin(d))
    return i, d[i]


def evaluate():
    models = sorted(p.name for p in (OUT / "pred").iterdir())
    res = {}
    for m in models:
        res[m] = {}
        for ds in ("panoptic", "3dpw"):
            if any((OUT / "frames").glob(f"{ds}_*")):
                res[m].update(gt_metrics(m, ds))
        carda(m, res[m])
        r = res[m]
        print(f"{m:12s} " + " ".join(f"{k} {v:.4f}" for k, v in r.items() if v is not None))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    import pandas as pd
    pd.DataFrame(res).T.rename_axis("model").to_csv(OUT / "summary_by_model.csv", float_format="%.4f")


def gt_metrics(m, ds):
    """Accuracy and temporal error against ground truth for one dataset (keys prefixed with ds)."""
    nerr, found, vel_err, jit_pred, jit_gt = [], [], [], [], []
    for clip in sorted((OUT / "frames").glob(f"{ds}_*")):
        gt = np.load(clip / "gt.npz")
        pr = np.load(OUT / "pred" / m / f"{clip.name}.npz")
        frames = sorted({int(k.split("/")[0]) for k in gt.files})
        tracks = {}
        for i in frames:
            k = pr[f"{i:08d}/k"]
            for pid in sorted({int(x.split("/")[1]) for x in gt.files if x.startswith(f"{i}/")}):
                uv, vis = gt[f"{i}/{pid}/uv"], gt[f"{i}/{pid}/vis"]
                diag = np.hypot(*np.ptp(uv[vis], axis=0))
                j, dist = match(k, uv, vis)
                ok = j is not None and dist / diag < 0.1
                found.append(ok)
                if ok:
                    nerr += list(np.linalg.norm(k[j][:17][vis] - uv[vis], axis=1) / diag)
                    tracks.setdefault(pid, {})[i] = (k[j][:17], uv, vis, diag)
        for pid, tr in tracks.items():          # temporal error on runs of consecutive frames
            for i in tr:
                if i - 1 in tr and i + 1 in tr:
                    (p0, g0, v0, _), (p1, g1, v1, dg), (p2, g2, v2, _) = tr[i - 1], tr[i], tr[i + 1]
                    v = v0 & v1 & v2
                    if v.any():
                        vel_err += list(np.linalg.norm((p1 - p0)[v] - (g1 - g0)[v], axis=1) / dg)
                        jit_pred += list(np.linalg.norm((p2 - 2 * p1 + p0)[v], axis=1) / dg)
                        jit_gt += list(np.linalg.norm((g2 - 2 * g1 + g0)[v], axis=1) / dg)
    e = np.array(nerr)
    return {f"{ds}_people_found": float(np.mean(found)), f"{ds}_pck05": float((e <= 0.05).mean()),
            f"{ds}_pck10": float((e <= 0.10).mean()), f"{ds}_nerr_median": float(np.median(e)),
            f"{ds}_velocity_error": float(np.mean(vel_err)), f"{ds}_jitter": float(np.mean(jit_pred)),
            f"{ds}_jitter_ground_truth": float(np.mean(jit_gt))}


def carda(m, res_m):
    """CarDA: no ground truth. Track the worker (largest person, then the nearest pose; a jump of
    more than half the box diagonal breaks the track) and report, per camera, the share of frames
    with the worker found and the median jitter. Degenerate poses (box diagonal < 20 px, e.g.
    RTMO's all-zero placeholder when nobody is detected) count as no detection."""
    for view in ("in", "out"):
        jit, seen = [], []
        for clip in sorted((OUT / "frames").glob(f"carda_*_{view}_camera*")):
            pr = np.load(OUT / "pred" / m / f"{clip.name}.npz")
            track, prev = [], None
            for n in sorted({x.split("/")[0] for x in pr.files}):
                k = [p for p in pr[f"{n}/k"][:, :17] if np.hypot(*np.ptp(p, axis=0)) >= 20]
                j = None
                if k and prev is None:
                    j = int(np.argmax([np.prod(np.ptp(p, axis=0)) for p in k]))
                elif k:
                    d = [np.median(np.linalg.norm(p - prev, axis=1)) for p in k]
                    j = int(np.argmin(d)) if min(d) <= 0.5 * np.hypot(*np.ptp(prev, axis=0)) else None
                prev = k[j] if j is not None else None
                track.append(prev)
            seen += [t is not None for t in track]
            for a, b, c in zip(track, track[1:], track[2:]):
                if a is not None and b is not None and c is not None:
                    jit.append(np.median(np.linalg.norm(c - 2 * b + a, axis=1)) / np.hypot(*np.ptp(b, axis=0)))
        res_m[f"carda_{view}_found"] = float(np.mean(seen)) if seen else None
        res_m[f"carda_{view}_jitter_median"] = float(np.median(jit)) if jit else None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "frames":
        panoptic_frames()
        carda_frames()
    elif cmd == "frames3dpw":
        threedpw_frames(sys.argv[2] if len(sys.argv) > 2 else "test")
    elif cmd == "infer" and len(sys.argv) == 3:
        infer(sys.argv[2])
    elif cmd == "eval":
        evaluate()
    else:
        sys.exit(__doc__)
