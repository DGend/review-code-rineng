# PWW-DISCLOSURE: public
"""Run DETRPose (real-time end-to-end transformer, arXiv 2506.13027) on the comparison images.

Needs the DETRPose repository (https://github.com/SebastianJanampa/DETRPose, Apache-2.0) at
$DETRPOSE_ROOT with its released weights in weights/, and a PyTorch environment for it.
Model loading and post-processing follow the repository's tools/inference/torch_inf.py.
DETRPose returns one score per person, which is used as every keypoint's confidence.

Usage: DETRPOSE_ROOT=~/third_party/DETRPose python topic/E/src/compare_pose_models_detrpose.py {l,x}
Writes topic/E/result/pose_model_comparison/detrpose_<size>.npz (same format as compare_pose_models_infer.py).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose.data_paths import IMG_DIR  # noqa: E402
from compare_pose_models_infer import OUT, image_list  # noqa: E402

KEEP = 0.3          # person score kept as a candidate; the labelled worker is matched afterwards


def load(size, device):
    repo = Path(os.environ["DETRPOSE_ROOT"]).expanduser()
    sys.path.insert(0, str(repo))
    from src.core import LazyConfig, instantiate
    cfg = LazyConfig.load(str(repo / f"configs/detrpose/detrpose_hgnetv2_{size}.py"))
    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False
    model, post = instantiate(cfg.model), instantiate(cfg.postprocessor)
    ck = torch.load(repo / f"weights/detrpose_hgnetv2_{size}.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.post = model.deploy(), post.deploy()

        def forward(self, x, size):
            return self.post(self.model(x), size)

    return Deployed().to(device).eval()


@torch.no_grad()
def main(size):
    out = OUT / f"detrpose_{size}.npz"
    if out.exists():
        sys.exit(f"{out} exists; delete it to re-run")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tf = load(size, device), T.Compose([T.Resize((640, 640)), T.ToTensor()])
    store, stems, t0 = {}, image_list(), time.time()
    for i, stem in enumerate(stems):
        im = Image.open(IMG_DIR / f"{stem}.jpg").convert("RGB")
        scores, _labels, kpts = model(tf(im).unsqueeze(0).to(device), torch.tensor([list(im.size)]).to(device))
        sc, kp = scores[0].cpu().numpy(), kpts[0].cpu().numpy()[..., :2]
        keep = sc > KEEP
        store[f"{stem}/k"] = kp[keep].astype(np.float32)
        store[f"{stem}/s"] = np.repeat(sc[keep, None], kp.shape[1], axis=1).astype(np.float32)
        if i % 200 == 0:
            print(f"detrpose_{size}: {i}/{len(stems)}, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    print(f"detrpose_{size}: {len(stems)} images in {time.time() - t0:.0f} s on {device} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("l", "x"):
        sys.exit(__doc__)
    main(sys.argv[1])
