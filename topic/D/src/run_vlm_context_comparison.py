# PWW-DISCLOSURE: public
"""Run heuristic vs. HuggingFace-VLM context extraction on a stratified
50-image sample and compute internal-consistency agreement, to give
manuscript Section 4.7's "50 images, qualitative" claim a real, traceable
result (previously unexecuted per topic/A/experiment-queue.md exp-004-vlm-gt).

Reads from topic/A's declared batch_v1 pose cache and the raw Busan HSE
images. Writes metrics.json + per_image.csv under
PWW_RESULT_DIR (topic/D/result/exp-vlm50-context-comparison/).
"""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BATCH_DIR = ROOT / "ace_pose/_runs/batch_v1"
import sys
sys.path.insert(0, str(ROOT))
from ace_pose.data_paths import IMG_DIR
# The published per_image.csv was produced from an earlier JPEG export of the same
# images (identical content, mean pixel difference below 0.3 of 255).
IMAGE_DIRS = [IMG_DIR]
RESULT_DIR = Path(os.environ.get(
    "PWW_RESULT_DIR",
    str(ROOT / "topic/D/result/exp-vlm50-context-comparison"),
))

import sys
sys.path.insert(0, str(ROOT))
from ace_pose.major_joints import Major25Pose
from ace_pose.vlm_context import ContextExtractor, HeuristicBackend, HuggingFaceVLMBackend


def sample_50(seed=13):
    df = pd.read_csv(BATCH_DIR / "summary.csv")
    df = df[df["status"] == "ok"].reset_index(drop=True)
    rng = random.Random(seed)
    per_domain = {"manufacturing": 25, "eldercare": 15, "logistics": 10}
    rows = []
    for domain, n in per_domain.items():
        pool = df[df["domain"] == domain]["file_stem"].tolist()
        rows.extend(rng.sample(pool, min(n, len(pool))))
    return rows


def load_pose(file_stem):
    data = np.load(BATCH_DIR / "details" / f"{file_stem}.npz")
    return Major25Pose(
        keypoints=data["kpts_m25"].astype(np.float64),
        confidence=data["conf_m25"].astype(np.float64),
        source=np.zeros(25, dtype=np.int8),
    )


def load_image(file_stem):
    for image_dir in IMAGE_DIRS:
        path = image_dir / f"{file_stem}.jpg"
        if path.exists():
            return cv2.imread(str(path))
    return None


def main():
    stems = sample_50()
    heuristic = ContextExtractor(HeuristicBackend())
    vlm = ContextExtractor(HuggingFaceVLMBackend())

    rows = []
    vlm_unavailable = False
    for i, stem in enumerate(stems):
        pose = load_pose(stem)
        img = load_image(stem)
        h_ctx = heuristic(pose=pose)
        v_ctx = vlm(image_bgr=img, pose=pose)
        if v_ctx.backend != "hf-vlm":
            vlm_unavailable = True
        rows.append({
            "file_stem": stem,
            "h_load_band": h_ctx.load_band,
            "v_load_band": v_ctx.load_band,
            "h_coupling": h_ctx.coupling,
            "v_coupling": v_ctx.coupling,
            "h_static": h_ctx.activity_static,
            "v_static": v_ctx.activity_static,
            "h_repetitive": h_ctx.activity_repetitive,
            "v_repetitive": v_ctx.activity_repetitive,
            "h_rapid": h_ctx.activity_rapid,
            "v_rapid": v_ctx.activity_rapid,
            "v_backend_actual": v_ctx.backend,
            "v_confidence": v_ctx.confidence,
        })
        print(f"[{i+1}/{len(stems)}] {stem} v_backend={v_ctx.backend}", flush=True)

    df = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / "per_image.csv", index=False)

    n = len(df)
    metrics = {
        "n_images": n,
        "domains_sampled": {"manufacturing": 25, "eldercare": 15, "logistics": 10},
        "vlm_backend_fell_back_to_heuristic": bool(vlm_unavailable),
        "vlm_model_id": vlm.backend.model_id if hasattr(vlm.backend, "model_id") else None,
        "load_band_agreement_rate": float((df["h_load_band"] == df["v_load_band"]).mean()),
        "coupling_agreement_rate": float((df["h_coupling"] == df["v_coupling"]).mean()),
        "activity_static_agreement_rate": float((df["h_static"] == df["v_static"]).mean()),
        "activity_repetitive_agreement_rate": float((df["h_repetitive"] == df["v_repetitive"]).mean()),
        "activity_rapid_agreement_rate": float((df["h_rapid"] == df["v_rapid"]).mean()),
        "mean_vlm_self_reported_confidence": float(df["v_confidence"].mean()),
    }
    with open(RESULT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
