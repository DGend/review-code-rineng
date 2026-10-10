# PWW-DISCLOSURE: public
"""Compare pose models on keypoint accuracy and on the action levels they lead to.

For every model and image, the detection that best matches the labelled worker is taken
(label_distance, as in batch_infer.py). Then:
  keypoints   normalised error (distance / label-box diagonal) and PCK at 0.05 and 0.10 on
              the 15 body joints the annotation has (no ears), and the share of images where
              the labelled worker is found (match distance <= LABEL_MATCH_MAX);
  downstream  OWAS/RULA/REBA action levels from each model's 2D pose with the same scorer,
              task context and joint set, against the levels from the annotated keypoints
              (evaluation images) and against the expert (22 pilot images).
Models: the current RTMW3D-x (cached poses) and every <model>.npz written by
compare_pose_models_infer.py or the DETRPose runner.

Usage: python topic/E/src/compare_pose_models_eval.py
Writes topic/E/result/pose_model_comparison/summary.json and per_joint.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose.batch_infer import COCO_TO_LABEL, LABEL_MATCH_MAX, label_distance  # noqa: E402
from ace_pose.data_paths import JSON_DIR  # noqa: E402
from ace_pose.major_joints import Major25Pose  # noqa: E402
from eval_action_agreement import DROPPED, LEVELS, label_pose, levels_of, metrics, with_neck_pelvis  # noqa: E402

OUT = ROOT / "topic/E/result/pose_model_comparison"
BATCH = ROOT / "ace_pose/_runs/batch_v1"
JOINTS = {c: l for c, l in COCO_TO_LABEL.items()}          # 15 body joints with annotations
NAMES = ["nose", "eye_l", "eye_r", "ear_l", "ear_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
         "wrist_l", "wrist_r", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"]


def ann_of(row):
    kp = np.array([[row[f"keypoint_{i}_x"], row[f"keypoint_{i}_y"], row[f"keypoint_{i}_v"]] for i in range(27)])
    return {"bbox": [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2], "keypoints": kp}


def m25_from_2d(k, s):
    """Major-25 pose from COCO-17 (+ 6 foot points) image coordinates; the joint set is the
    annotation's (ears, small toes and heels dropped), so every model is scored alike."""
    kp, conf = np.zeros((25, 3)), np.zeros(25)
    n = min(len(k), 23)
    kp[:n, :2], conf[:n] = k[:n], np.clip(s[:n], 0, 1)
    conf[DROPPED] = 0.0
    return with_neck_pelvis(kp, conf)


def models() -> dict:
    out = {"rtmw3d_current": None}
    for p in sorted(OUT.glob("*.npz")):
        out[p.stem] = np.load(p)
    return out


def detections(name, data, stem):
    if name == "rtmw3d_current":
        z = np.load(BATCH / "details" / f"{stem}.npz")
        return z["kpts2d_133"][None, :23].astype(float), z["scores_133"][None, :23].astype(float)
    if f"{stem}/k" not in data:
        return None, None
    return data[f"{stem}/k"].astype(float), data[f"{stem}/s"].astype(float)


def main():
    summ = pd.read_csv(BATCH / "summary.csv").set_index("file_stem")
    lab = pd.read_csv(Path(JSON_DIR).parent / "data_v5.csv")
    lab["file_stem"] = lab.file_name.str[:-4]
    lab = lab.drop_duplicates(["file_stem", "worker"], keep=False).set_index(["file_stem", "worker"])
    test = pd.read_csv(ROOT / "topic/E/result/action_agreement/test_split_20.csv").file_stem.tolist()
    pil = pd.read_csv(ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv")
    expert = pil[pil.rater == "expert"].set_index(["file_stem", "standard"]).action

    summary, joint_rows = {}, []
    for name, data in models().items():
        err, found, lev = [], [], {}
        for stem in dict.fromkeys(test + pil.file_stem.unique().tolist()):
            key = (stem, int(summ.at[stem, "worker_id"]))
            if key not in lab.index:
                continue
            row = lab.loc[key]
            ann = ann_of(row)
            k, s = detections(name, data, stem)
            if k is None or len(k) == 0:
                found.append(False)
                continue
            d = [label_distance(k[p, :17], ann) for p in range(len(k))]
            p = int(np.argmin(d))
            found.append(d[p] <= LABEL_MATCH_MAX)
            diag = max(np.hypot(row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1), 1.0)
            for c, l in JOINTS.items():
                if ann["keypoints"][l, 2] > 0:
                    err.append((NAMES[c], float(np.hypot(*(k[p, c] - ann["keypoints"][l, :2])) / diag)))
            label = summ.at[stem, "process_label"]
            lev[stem] = (levels_of(label_pose(row.to_dict()), label), levels_of(m25_from_2d(k[p], s[p]), label))
        e = pd.DataFrame(err, columns=["joint", "nerr"])
        res = {"images": len(found), "worker_found": float(np.mean(found)),
               "nerr_median": float(e.nerr.median()), "nerr_mean": float(e.nerr.mean()),
               "pck05": float((e.nerr <= 0.05).mean()), "pck10": float((e.nerr <= 0.10).mean())}
        tst = [st for st in test if st in lev]
        pst = [st for st in pil.file_stem.unique() if st in lev]
        for std, lv in LEVELS.items():
            res[f"labels_{std}"] = metrics([lev[st][0][std] for st in tst], [lev[st][1][std] for st in tst], lv)
            res[f"expert_{std}"] = metrics([int(expert[(st, std)]) for st in pst], [lev[st][1][std] for st in pst], lv)
        summary[name] = res
        for j, g in e.groupby("joint"):
            joint_rows.append({"model": name, "joint": j, "median_nerr": g.nerr.median(), "pck05": (g.nerr <= 0.05).mean()})
        print(f"{name:16s} found {res['worker_found']:.3f}  nerr med {res['nerr_median']:.4f}  PCK@.05 {res['pck05']:.3f}  "
              f"PCK@.10 {res['pck10']:.3f} | labels acc O/R/R "
              + "/".join(f"{res[f'labels_{s}']['accuracy']:.2f}" for s in LEVELS)
              + " | expert exact " + "/".join(f"{round(res[f'expert_{s}']['accuracy'] * res[f'expert_{s}']['n'])}" for s in LEVELS)
              + f" of {res['expert_OWAS']['n']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(joint_rows).to_csv(OUT / "per_joint.csv", index=False)


if __name__ == "__main__":
    main()
