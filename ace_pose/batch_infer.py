"""Batch inference: RTMW3D + major-25 filter + ergo_v2 over all 6,007 images.

Output: a parquet file per image with full keypoints, confidence, ergo
sub-scores, and metadata. Also a summary CSV with one row per image
suitable for direct analysis.

Run with:
    LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH \\
    python -m ace_pose.batch_infer
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.major_joints import coco_wholebody_to_major25, MAJOR25_NAMES
from ace_pose.ergo_v2.context_policy import cohort_context  # posture only, see context_policy
from ace_pose.ergo_v2 import score_all
from ace_pose.ergo_v2.result import ErgoResult


log = logging.getLogger("ace_pose.batch_infer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


from ace_pose.data_paths import DATA_ROOT
IMG_DIR = DATA_ROOT / "new_images"
JSON_DIR = DATA_ROOT / "new_json"
OUT_DIR = PROJECT_ROOT / "ace_pose" / "_runs" / "batch_v1"


# ----------------------------------------------------------------------
# Filename parsing
# ----------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^F(?P<factory>\d+)_P(?P<process>\d+)_A(?P<action>\d+)_W(?P<worker>\d+)_"
    r"D(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(?P<frame>\d+)$"
)


def parse_filename(stem: str) -> dict | None:
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    return {
        "file_stem": stem,
        "factory_id": int(m.group("factory")),
        "process_id": int(m.group("process")),
        "action_id": int(m.group("action")),
        "worker_id": int(m.group("worker")),
        "capture_dt": m.group("date"),
        "frame_idx": int(m.group("frame")),
    }


# Factory→domain mapping (per project metadata).
FACTORY_DOMAIN = {
    1: "manufacturing",
    2: "manufacturing",
    3: "manufacturing",       # SUNTECH
    4: "manufacturing",
    5: "manufacturing",
    6: "eldercare",
    7: "logistics",
}

# Process-id → human-readable Korean label (per project PDF).
# Where a number is unverified the label is best-effort and prefixed `?`.
PROCESS_LABEL = {
    1: "사상1", 2: "사상2", 3: "사상3", 4: "사상4", 5: "사상5",
    6: "사상6", 7: "용접1", 8: "용접2", 9: "용접3", 10: "용접4",
    11: "주조", 12: "벤딩", 13: "절단", 14: "조립", 15: "가공",
    16: "선별", 17: "도색", 18: "이동", 19: "도움", 20: "목욕수행",
    21: "씻기", 22: "마사지", 23: "식사",
}


# ----------------------------------------------------------------------
# Pose model singleton
# ----------------------------------------------------------------------

class PoseModel:
    """Lazy-init wrapper around the RTMW3D wholebody model."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from rtmlib import Wholebody3d
            # silence the noisy print() the rtmlib does on init
            buf = StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                cls._instance = Wholebody3d(
                    mode="balanced", backend="onnxruntime", device="cuda",
                )
        return cls._instance


def infer_one(image_bgr: np.ndarray):
    """Run RTMW3D on one image and return (kpts3d, scores, kpts2d)."""
    model = PoseModel.get()
    # rtmlib prints per-frame; capture and discard
    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        kpts3d, scores, _simcc, kpts2d = model(image_bgr)
    return kpts3d, scores, kpts2d


# ----------------------------------------------------------------------
# Per-image processing
# ----------------------------------------------------------------------

# Dataset label (27 keypoints) index for each COCO body keypoint, derived from
# single-annotation images; ears and feet have no unique label counterpart.
COCO_TO_LABEL = {0: 3, 1: 1, 2: 2, 5: 5, 6: 11, 7: 6, 8: 12, 9: 7, 10: 13,
                 11: 19, 12: 23, 13: 20, 14: 24, 15: 21, 16: 25}
_LR_SWAP = {1: 2, 2: 1, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11,
            13: 14, 14: 13, 15: 16, 16: 15}
# Median keypoint distance / label-box diagonal below which a detection is the
# labelled person (the distribution is bimodal with a gap above this value).
LABEL_MATCH_MAX = 0.05


def label_distance(kpts2d_body: np.ndarray, ann: dict) -> float:
    """Median 2D distance between a detection's COCO body keypoints and a labelled
    person's keypoints, divided by the label box diagonal (left/right swap allowed)."""
    lbl = np.asarray(ann.get("keypoints", []), dtype=float)
    if lbl.ndim != 2 or len(lbl) < 26:
        return float("inf")
    x1, y1, x2, y2 = ann["bbox"]          # corner format
    diag = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
    best = float("inf")
    for perm in (COCO_TO_LABEL, {c: COCO_TO_LABEL[_LR_SWAP.get(c, c)] for c in COCO_TO_LABEL}):
        pairs = [(c, l) for c, l in perm.items() if lbl[l, 2] > 0]
        if pairs:
            d = np.median([np.hypot(*(kpts2d_body[c] - lbl[l, :2])) for c, l in pairs]) / diag
            best = min(best, float(d))
    return best


def _pick_primary_person(
    kpts2d: np.ndarray, scores: np.ndarray, annotations: list[dict] | None,
    file_worker: int | None = None,
) -> tuple[int, dict | None, float]:
    """Return (person index, matched annotation, match distance).

    Detections are compared with the labelled worker in 2D image coordinates
    (the 3D keypoints are in per-person crop space and cannot be compared with
    image-space labels). The labelled worker is the file name's worker when it
    is among the labels, otherwise the first annotation: at F1-F5 the first
    annotation is the file-name worker in every multi-person image, and at
    F6/F7 the file-name token is a recording session. Without labels the most
    confident detection is returned.
    """
    if kpts2d.shape[0] == 0:
        return -1, None, float("inf")
    anns = annotations or []
    own = [a for a in anns if file_worker is not None and a.get("worker") == file_worker]
    anns = own or anns[:1]
    if not anns:
        return int(np.argmax(scores.mean(axis=1))), None, float("inf")
    d = np.array([[label_distance(k[:17], a) for a in anns] for k in kpts2d])
    p, a = np.unravel_index(int(np.argmin(d)), d.shape)
    return int(p), anns[a], float(d[p, a])


def process_image(img_path: Path, json_path: Path | None = None) -> dict:
    """Run the full RTMW3D → major-25 → ergo_v2 pipeline on one image."""
    record: dict = parse_filename(img_path.stem) or {"file_stem": img_path.stem}
    record["domain"] = FACTORY_DOMAIN.get(record.get("factory_id", 0), "unknown")
    record["process_label"] = PROCESS_LABEL.get(record.get("process_id", 0), "?")
    record["status"] = "ok"
    record["error"] = ""

    img = cv2.imread(str(img_path))
    if img is None:
        record["status"] = "read_failed"
        return record
    record["img_h"], record["img_w"] = img.shape[:2]

    annotations = None
    if json_path is not None and json_path.exists():
        try:
            with open(json_path) as f:
                annotations = json.load(f)["annotations"]
        except Exception as ex:
            record["error"] = f"json_read: {ex}"

    try:
        kpts3d, scores, kpts2d = infer_one(img)
    except Exception as ex:
        record["status"] = "inference_failed"
        record["error"] = f"{type(ex).__name__}: {ex}"
        return record

    if kpts3d.shape[0] == 0:
        record["status"] = "no_detection"
        return record

    # The file name's W token is a worker at F1-F5 but a recording session at
    # F6/F7; the matched label supplies the actual worker id.
    record["session_id"] = record.get("worker_id")
    primary, ann, match_d = _pick_primary_person(kpts2d, scores, annotations, record.get("worker_id"))
    record["n_detections"] = int(kpts3d.shape[0])
    record["primary_idx"] = int(primary)
    record["label_match_dist"] = match_d
    record["label_match_ok"] = bool(match_d <= LABEL_MATCH_MAX)
    if ann is not None:
        record["worker_id"] = int(ann["worker"])

    primary_kpts = kpts3d[primary]            # (133, 3)
    primary_scores = scores[primary]          # (133,)
    primary_2d = kpts2d[primary]              # (133, 2)

    # Major-25 filter
    m25 = coco_wholebody_to_major25(primary_kpts, primary_scores)

    # ergo_v2
    try:
        result: ErgoResult = score_all(
            m25,
            world_up=np.array([0.0, -1.0, 0.0]),
            walking=False,
            **cohort_context(record["process_label"]),
            domain=record["domain"],
            process_id=record.get("process_id"),
            process_label=record["process_label"],
        )
    except Exception as ex:
        record["status"] = "scoring_failed"
        record["error"] = f"{type(ex).__name__}: {ex}"
        return record

    record.update(ergo_columns(result))
    record.update({
        # Pose-confidence summary stats
        "kpt_conf_mean": float(np.mean(primary_scores)),
        "kpt_conf_min": float(np.min(primary_scores)),
        "kpt_conf_max": float(np.max(primary_scores)),
        "m25_conf_mean": float(np.mean(m25.confidence)),
    })

    # Also stash detail blobs in a side dict (saved per-image to parquet)
    record["_detail"] = {
        "kpts3d_133": primary_kpts.astype(np.float32),
        "scores_133": primary_scores.astype(np.float32),
        "kpts2d_133": primary_2d.astype(np.float32),
        "kpts_m25": m25.keypoints.astype(np.float32),
        "conf_m25": m25.confidence.astype(np.float32),
    }
    return record


def ergo_columns(result: ErgoResult) -> dict:
    """Flatten an ErgoResult into the summary-CSV score columns."""
    return {
        "unmeasurable": ";".join(result.extras.get("unmeasurable", [])),
        "owas_back": result.owas.back,
        "owas_arms": result.owas.arms,
        "owas_legs": result.owas.legs,
        "owas_load": result.owas.load,
        "owas_action": result.owas.action_category,
        "owas_trunk_flex_deg": result.owas.trunk_flex_deg,
        "owas_trunk_twist_deg": result.owas.trunk_twist_deg,
        "owas_trunk_lateral_deg": result.owas.trunk_lateral_deg,
        "owas_knee_l_deg": result.owas.knee_left_deg,
        "owas_knee_r_deg": result.owas.knee_right_deg,

        "rula_grand": result.rula.grand_score,
        "rula_action": result.rula.action_level,
        "rula_side": result.rula.side,
        "rula_upper_arm": result.rula.upper_arm,
        "rula_lower_arm": result.rula.lower_arm,
        "rula_wrist": result.rula.wrist,
        "rula_neck": result.rula.neck,
        "rula_trunk": result.rula.trunk,
        "rula_score_a": result.rula.score_a,
        "rula_score_b": result.rula.score_b,
        "rula_upper_arm_deg": result.rula.upper_arm_deg,
        "rula_lower_arm_deg": result.rula.lower_arm_deg,
        "rula_neck_deg": result.rula.neck_deg,
        "rula_trunk_deg": result.rula.trunk_deg,

        "reba_grand": result.reba.grand_score,
        "reba_action": result.reba.action_level,
        "reba_side": result.reba.side,
        "reba_trunk": result.reba.trunk,
        "reba_neck": result.reba.neck,
        "reba_legs": result.reba.legs,
        "reba_upper_arm": result.reba.upper_arm,
        "reba_lower_arm": result.reba.lower_arm,
        "reba_wrist": result.reba.wrist,
        "reba_score_a": result.reba.score_a,
        "reba_score_b": result.reba.score_b,
        "reba_activity": result.reba.activity_score,
        "reba_knee_deg": result.reba.knee_deg,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def iter_images(img_dir: Path, limit: int | None = None) -> Iterable[Path]:
    paths = sorted(img_dir.glob("*.jpg"))
    if limit:
        paths = paths[:limit]
    return paths


def run(limit: int | None = None, out_dir: Path = OUT_DIR):
    out_dir.mkdir(exist_ok=True, parents=True)
    detail_dir = out_dir / "details"
    detail_dir.mkdir(exist_ok=True)

    paths = list(iter_images(IMG_DIR, limit=limit))
    log.info("Batch run: %d images → %s", len(paths), out_dir)

    summary_rows: list[dict] = []
    t_start = time.time()
    last_log = t_start

    for i, p in enumerate(paths):
        rec = process_image(p, JSON_DIR / f"{p.stem}.json")
        detail = rec.pop("_detail", None)
        if detail is not None:
            np.savez_compressed(detail_dir / f"{p.stem}.npz", **detail)
        summary_rows.append(rec)

        now = time.time()
        if now - last_log > 5.0 or i == len(paths) - 1:
            done = i + 1
            rate = done / (now - t_start)
            eta = (len(paths) - done) / max(rate, 1e-6)
            log.info(
                "%4d/%d  %.1f img/s  ETA %.0fs  last=%s status=%s",
                done, len(paths), rate, eta, p.name, rec.get("status"),
            )
            last_log = now

    df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "summary.csv"
    parquet_path = out_dir / "summary.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as ex:
        log.warning("parquet save failed (%s); CSV only.", ex)

    # Lightweight stats
    log.info("=" * 60)
    log.info("Done in %.1fs.", time.time() - t_start)
    status_counts = df["status"].value_counts().to_dict()
    log.info("Status counts: %s", status_counts)
    if "owas_action" in df:
        ok = df[df["status"] == "ok"]
        log.info("OWAS action dist: %s", ok["owas_action"].value_counts().sort_index().to_dict())
        log.info("RULA action dist: %s", ok["rula_action"].value_counts().sort_index().to_dict())
        log.info("REBA action dist: %s", ok["reba_action"].value_counts().sort_index().to_dict())
    log.info("Summary CSV: %s", csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N images (smoke test)")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    run(limit=args.limit, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
