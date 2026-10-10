# PWW-DISCLOSURE: public
"""Run alternative pose models on the comparison images and cache every detected person.

Images: the frozen 20% evaluation list (eval_action_agreement_split.py) and the 22 expert
pilot images. For each image the 2D keypoints and scores of all detected people are stored,
COCO-17 body order first (whole-body models also keep their six foot points, 17-22).
The current pipeline's RTMW3D-x poses are not re-run: its cached poses are read directly by
compare_pose_models_eval.py.

Usage: python topic/E/src/compare_pose_models_infer.py MODEL
  MODEL: rtmw (RTMW-x whole-body, top-down), rtmpose (RTMPose-x body, top-down),
         rtmo (RTMO-l, one-stage multi-person), vitpose (ViTPose-B, top-down, YOLOX-x boxes)
Writes topic/E/result/pose_model_comparison/<MODEL>.npz (keys <stem>/k and <stem>/s).
Set BIOERA_DEVICE=cuda to run on a GPU (needs onnxruntime-gpu).
"""
import os
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
from ace_pose.data_paths import IMG_DIR  # noqa: E402

OUT = ROOT / "topic/E/result/pose_model_comparison"
RTMO_L = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
VITPOSE_B = "https://huggingface.co/onnx-community/vitpose-base-simple/resolve/main/onnx/model.onnx"


def image_list() -> list[str]:
    test = pd.read_csv(ROOT / "topic/E/result/action_agreement/test_split_20.csv").file_stem.tolist()
    pilot = pd.read_csv(ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv").file_stem.unique().tolist()
    return list(dict.fromkeys(test + pilot))


def build(name):
    import rtmlib
    device = os.environ.get("BIOERA_DEVICE", "cpu")      # "cuda" with onnxruntime-gpu
    kw = dict(backend="onnxruntime", device=device)
    if name == "rtmw":
        return rtmlib.Wholebody(mode="performance", **kw), 23
    if name == "rtmpose":
        return rtmlib.Body(mode="performance", **kw), 17
    if name == "rtmo":
        return rtmlib.RTMO(onnx_model=RTMO_L, model_input_size=(640, 640), **kw), 17
    if name == "vitpose":
        det = rtmlib.YOLOX(onnx_model=rtmlib.Body.MODE["performance"]["det"], model_input_size=(640, 640), **kw)
        pose = rtmlib.ViTPose(onnx_model=VITPOSE_B, model_input_size=(192, 256), device=device)
        return (lambda img: pose(img, bboxes=det(img))), 17
    sys.exit(f"unknown model {name}")


def main(name):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.npz"
    if out.exists():
        sys.exit(f"{out} exists; delete it to re-run")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        model, k = build(name)
    stems, store, t0 = image_list(), {}, time.time()
    for i, stem in enumerate(stems):
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            kp, sc = model(img)
        kp, sc = np.asarray(kp, float).reshape(-1, kp.shape[-2] if np.ndim(kp) == 3 else 0, 2), np.asarray(sc, float)
        store[f"{stem}/k"] = kp[:, :k].astype(np.float32)
        store[f"{stem}/s"] = sc.reshape(len(kp), -1)[:, :k].astype(np.float32)
        if i % 100 == 0:
            print(f"{name}: {i}/{len(stems)} images, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out, **store)
    print(f"{name}: {len(stems)} images in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
