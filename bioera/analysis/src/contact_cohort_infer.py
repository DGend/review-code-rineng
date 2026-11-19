# PWW-DISCLOSURE: public
"""Re-run RTMW3D on every cohort image and cache, per image, the 2D major-25
keypoints of the primary worker (matched to the batch_v1 pose) and of every
other detected person. This restores what batch_infer discarded, so the
second-person contact-load rule (ace_pose.vlm_context, Section 4.9) can be
evaluated cohort-wide by contact_cohort_score.py. Resumable: images already
cached are skipped.

Writes ace_pose/_runs/contact_cohort/cache/<stem>.npz (p2d, others2d, n_people,
match_err). Source: Busan AI-Hub images + ace_pose/_runs/batch_v1/details.
"""
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ace_pose.major_joints import coco_wholebody_to_major25

from ace_pose.data_paths import IMG_DIR
BATCH = ROOT / "ace_pose/_runs/batch_v1"
CACHE = ROOT / "ace_pose/_runs/contact_cohort/cache"


def to_2d(k2, sc):
    return coco_wholebody_to_major25(np.concatenate([k2, np.zeros((133, 1))], axis=1), sc).keypoints


def main():
    from rtmlib import Wholebody3d
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        model = Wholebody3d(mode="balanced", backend="onnxruntime", device="cpu")
    df = pd.read_csv(BATCH / "summary.csv")
    stems = df[df.status == "ok"].file_stem.tolist()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for i, stem in enumerate(stems):
        out = CACHE / f"{stem}.npz"
        if out.exists():
            continue
        cached = np.load(BATCH / "details" / f"{stem}.npz")["kpts_m25"].astype(np.float64)
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        if img is None:
            print(f"missing image {stem}", flush=True)
            continue
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            k3, sc, _simcc, k2 = model(img)
        if k3.shape[0] == 0:
            continue
        err = [float(np.nanmean(np.linalg.norm(coco_wholebody_to_major25(k3[j], sc[j]).keypoints - cached, axis=1)))
               for j in range(k3.shape[0])]
        p = int(np.argmin(err))
        others = [to_2d(k2[j], sc[j]) for j in range(k3.shape[0]) if j != p]
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, p2d=to_2d(k2[p], sc[p]),
                 others2d=np.array(others) if others else np.zeros((0, 25, 3)),
                 n_people=k3.shape[0], match_err=err[p])
        tmp.replace(out)
        done += 1
        if done % 250 == 0:
            rate = done / (time.time() - t0)
            print(f"{i+1}/{len(stems)}  {rate:.1f} img/s  ETA {(len(stems)-i-1)/rate/60:.0f} min", flush=True)
    print(f"DONE {len(list(CACHE.glob('*.npz')))} cached", flush=True)


if __name__ == "__main__":
    main()
