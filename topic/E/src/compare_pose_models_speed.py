# PWW-DISCLOSURE: public
"""Latency, throughput, memory and size of the compared pose models, and of the scoring layer.

Each model runs in its own process (one call per model) on images held in memory, so disk
reads are excluded; the time covers the model's own preprocessing, person detection, pose
estimation and postprocessing for one image (batch 1), after a warm-up. Peak GPU memory is
sampled from nvidia-smi for this process. The scoring layer (OWAS, RULA, REBA with task
context) is timed on cached poses.

Usage: python topic/E/src/compare_pose_models_speed.py MODEL DEVICE [N]
  MODEL: rtmw3d (current, balanced), rtmw, rtmpose, rtmo, vitpose, detrpose_l, detrpose_x, scoring
  DEVICE: cuda | cpu; N images (default 200 on cuda, 30 on cpu)
Appends one JSON line to topic/E/result/pose_model_comparison/speed.jsonl.
"""
import json
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose.data_paths import IMG_DIR  # noqa: E402

OUT = ROOT / "topic/E/result/pose_model_comparison"
SEED, WARMUP = 20261009, 10


def images(n):
    stems = pd.read_csv(ROOT / "topic/E/result/action_agreement/test_split_20.csv").file_stem.to_numpy()
    stems = np.random.default_rng(SEED).choice(stems, size=n, replace=False)
    return stems, [cv2.imread(str(IMG_DIR / f"{s}.jpg")) for s in stems]


def onnx_files(obj, depth=0):
    """ONNX files held by an rtmlib model (detector and pose), for the size column."""
    if hasattr(obj, "session"):
        p = str(getattr(obj, "onnx_model", ""))
        if not Path(p).is_file():   # a download URL: rtmlib caches the unpacked file by name
            hits = list((Path.home() / ".cache/rtmlib/hub/checkpoints").glob(Path(p).stem.split("?")[0] + "*.onnx"))
            p = str(hits[0]) if hits else ""
        return [p] if p else []
    if depth > 2 or not hasattr(obj, "__dict__"):
        return []
    return [q for v in vars(obj).values() for q in onnx_files(v, depth + 1)]


def build(name, device):
    import rtmlib
    kw = dict(backend="onnxruntime", device=device)
    if name == "rtmw3d":
        m = rtmlib.Wholebody3d(mode="balanced", **kw)
        return (lambda img: m(img)), m
    if name.startswith("detrpose"):
        import torch
        from compare_pose_models_detrpose import load
        import torchvision.transforms as T
        from PIL import Image
        mod, tf = load(name[-1], device), T.Compose([T.Resize((640, 640)), T.ToTensor()])

        @torch.no_grad()
        def run(img):
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out = mod(tf(im).unsqueeze(0).to(device), torch.tensor([list(im.size)]).to(device))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return out
        return run, mod
    os.environ["BIOERA_DEVICE"] = device
    from compare_pose_models_infer import build as b
    m, _ = b(name)
    return m, m


def model_size(name, obj):
    if name.startswith("detrpose"):
        n = sum(p.numel() for p in obj.parameters())
        f = Path(os.environ["DETRPOSE_ROOT"]).expanduser() / f"weights/detrpose_hgnetv2_{name[-1]}.pth"
        return {"params_M": n / 1e6, "file_MB": f.stat().st_size / 1e6}
    files = onnx_files(obj)
    return {"onnx_files": [Path(p).name for p in files], "file_MB": sum(Path(p).stat().st_size for p in files) / 1e6}


def gpu_used_mb():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5).stdout
    return float(q.split()[0])


class GpuPeak(threading.Thread):
    """Peak whole-GPU memory in use; under WSL nvidia-smi cannot attribute memory to a process,
    so the model's footprint is this peak minus the reading taken before the model was loaded."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0.0, False

    def run(self):
        while not self.stop:
            try:
                self.peak = max(self.peak, gpu_used_mb())
            except Exception:
                pass
            time.sleep(0.2)


def time_scoring(n):
    from ace_pose.ergo_v2.context_policy import task_context
    from ace_pose.ergo_v2.pipeline import score_all
    from ace_pose.major_joints import Major25Pose
    summ = pd.read_csv(ROOT / "ace_pose/_runs/batch_v1/summary.csv").set_index("file_stem")
    stems, _ = images(1)
    stems = pd.read_csv(ROOT / "topic/E/result/action_agreement/test_split_20.csv").file_stem[:n]
    poses = []
    for s in stems:
        z = np.load(ROOT / f"ace_pose/_runs/batch_v1/details/{s}.npz")
        poses.append((Major25Pose(keypoints=z["kpts_m25"].astype(float), confidence=z["conf_m25"].astype(float),
                                  source=np.zeros(25, np.int8)), task_context(summ.at[s, "process_label"])))
    up = np.array([0.0, -1.0, 0.0])
    t = []
    for p, c in poses:
        t0 = time.perf_counter()
        score_all(p, world_up=up, walking=False, **c)
        t.append(time.perf_counter() - t0)
    return np.array(t[WARMUP:])


def main(name, device, n):
    if name == "scoring":
        t = time_scoring(n + WARMUP)
        res = {"model": "scoring (OWAS+RULA+REBA)", "device": "cpu"}
    else:
        stems, imgs = images(n + WARMUP)
        base = gpu_used_mb() if device.startswith("cuda") else None
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            run, obj = build(name, device)
        gpu = GpuPeak()
        if device.startswith("cuda"):
            gpu.start()
        t = []
        for i, img in enumerate(imgs):
            t0 = time.perf_counter()
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run(img)
            if i >= WARMUP:
                t.append(time.perf_counter() - t0)
        gpu.stop = True
        t = np.array(t)
        res = {"model": name, "device": device, **model_size(name, obj),
               "gpu_mem_MB": gpu.peak - base if device.startswith("cuda") else None}
    res.update({"n": int(len(t)), "latency_ms_median": float(np.median(t) * 1e3),
                "latency_ms_p90": float(np.percentile(t, 90) * 1e3), "throughput_fps": float(1 / t.mean()),
                "cpu": platform.processor() or platform.machine(), "threads": os.cpu_count()})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "speed.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dev = sys.argv[2]
    main(sys.argv[1], "cuda:0" if dev == "cuda" and sys.argv[1].startswith("detrpose") else dev,
         int(sys.argv[3]) if len(sys.argv) > 3 else (200 if dev == "cuda" else 30))
