# PWW-DISCLOSURE: public
"""Qualitative comparison samples: three annotated evaluation images per workplace (F1-F7),
one folder per image, with the annotated keypoints and each model's keypoints drawn.

Images come from the frozen 20% evaluation list and have an annotation of the analysed
worker. Per workplace, three images from different recording sessions (different processes
first) are drawn with a fixed seed. Every face is pixelated before any keypoint is drawn:
the worker's from the RTMW3D face landmarks, everyone else's from the head keypoints of the
whole-body model. Each folder holds 00_label.jpg, one image per model, and grid.jpg.

Usage: python topic/E/src/compare_pose_models_samples.py
Writes topic/E/result/pose_model_comparison/samples/ (folders, index.csv).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_pose.batch_infer import COCO_TO_LABEL, label_distance  # noqa: E402
from ace_pose.data_paths import IMG_DIR, JSON_DIR  # noqa: E402
from ace_pose.run_pose_visualization import mosaic_face  # noqa: E402
from compare_pose_models_eval import BATCH, OUT, ann_of, detections, models  # noqa: E402

SEED, PER_SITE = 20261009, 3
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
LABELS = {"label": "Annotation", "rtmw3d_current": "RTMW3D-x (ours)", "rtmw": "RTMW-x", "rtmpose": "RTMPose-x",
          "rtmo": "RTMO-l", "vitpose": "ViTPose-B", "detrpose_l": "DETRPose-L", "detrpose_x": "DETRPose-X"}


def pick(per, summ):
    m = per.merge(summ[["file_stem", "factory_id", "process_label"]], on="file_stem")
    m["session"] = m.file_stem.str.rsplit("_", n=1).str[0]
    rng, out = np.random.default_rng(SEED), []
    for f, g in m.groupby("factory_id"):
        g = g.iloc[rng.permutation(len(g))]
        g = g.drop_duplicates("session")                        # different recording sessions
        g = pd.concat([g.drop_duplicates("process_label"), g]).drop_duplicates("file_stem")  # processes first
        out.append(g.head(PER_SITE))
    return pd.concat(out)


def blur_heads(img, kps, scs, block=12):
    """Pixelate each detected person's head from nose, eyes and ears (scaled by shoulder width)."""
    h, w = img.shape[:2]
    for k, s in zip(kps, scs):
        head = k[:5][s[:5] > 0.1]
        if len(head) == 0:
            continue
        sh = np.linalg.norm(k[5] - k[6]) if min(s[5], s[6]) > 0.1 else 0.0
        half = max(np.ptp(head[:, 0]), np.ptp(head[:, 1]), 0.6 * sh, 20) * 0.9
        cx, cy = head.mean(axis=0)
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
        if x1 > x0 + 4 and y1 > y0 + 4:
            r = img[y0:y1, x0:x1]
            small = cv2.resize(r, (max(1, r.shape[1] // block), max(1, r.shape[0] // block)))
            img[y0:y1, x0:x1] = cv2.resize(small, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img


def draw(img, k, color, vis=None):
    t = max(2, img.shape[0] // 300)
    for a, b in EDGES:
        if vis is None or (vis[a] and vis[b]):
            cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), color, t, cv2.LINE_AA)
    for i in range(17):
        if vis is None or vis[i]:
            cv2.circle(img, tuple(map(int, k[i])), t + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(map(int, k[i])), t + 1, color, -1, cv2.LINE_AA)
    return img


def main():
    summ = pd.read_csv(BATCH / "summary.csv")
    per = pd.read_csv(OUT / "per_image_labels.csv") if (OUT / "per_image_labels.csv").exists() else \
        pd.read_csv(ROOT / "topic/E/result/action_agreement/per_image_labels.csv")
    test = set(pd.read_csv(ROOT / "topic/E/result/action_agreement/test_split_20.csv").file_stem)
    sel = pick(per[per.file_stem.isin(test)], summ)
    lab = pd.read_csv(Path(JSON_DIR).parent / "data_v5.csv")
    lab["file_stem"] = lab.file_name.str[:-4]
    lab = lab.drop_duplicates(["file_stem", "worker"], keep=False).set_index(["file_stem", "worker"])
    worker = summ.set_index("file_stem").worker_id
    mods, rtmw = models(), np.load(OUT / "rtmw.npz")
    root = OUT / "samples"
    rows = []
    for r in sel.itertuples():
        stem, row = r.file_stem, lab.loc[(r.file_stem, int(worker[r.file_stem]))]
        ann = ann_of(row)
        base = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        z = np.load(BATCH / "details" / f"{stem}.npz")
        base = mosaic_face(base, z["kpts2d_133"], z["scores_133"])
        base = blur_heads(base, rtmw[f"{stem}/k"], rtmw[f"{stem}/s"])
        x1, y1, x2, y2 = row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2
        m = 0.25 * max(x2 - x1, y2 - y1)
        cx0, cy0 = int(max(0, x1 - m)), int(max(0, y1 - m))
        cx1, cy1 = int(min(base.shape[1], x2 + m)), int(min(base.shape[0], y2 + m))
        folder = root / f"F{int(r.factory_id)}_{stem}"
        folder.mkdir(parents=True, exist_ok=True)
        panels = []
        # annotation, in COCO-17 order (ears are not annotated)
        lk, lv = np.zeros((17, 2)), np.zeros(17, bool)
        for c, l in COCO_TO_LABEL.items():
            lk[c], lv[c] = ann["keypoints"][l, :2], ann["keypoints"][l, 2] > 0
        out = draw(base.copy(), lk, (0, 200, 0), lv)[cy0:cy1, cx0:cx1]
        cv2.imwrite(str(folder / "00_label.jpg"), out)
        panels.append(("label", out))
        for i, name in enumerate(LABELS):
            if name == "label" or name not in mods:
                continue
            k, s = detections(name, mods[name], stem)
            if k is None or len(k) == 0:
                continue
            p = int(np.argmin([label_distance(k[j, :17], ann) for j in range(len(k))]))
            out = draw(base.copy(), k[p, :17], (40, 80, 230))[cy0:cy1, cx0:cx1]
            cv2.imwrite(str(folder / f"{i:02d}_{name}.jpg"), out)
            panels.append((name, out))
        h = 480
        tiles = []
        for name, im in panels:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
            cv2.rectangle(im, (0, 0), (im.shape[1], 34), (255, 255, 255), -1)
            cv2.putText(im, LABELS[name], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(im)
        wmax = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(folder / "grid.jpg"), grid)
        rows.append({"folder": folder.name, "file_stem": stem, "factory": f"F{int(r.factory_id)}",
                     "process": r.process_label, "domain": r.domain})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    print(f"{len(rows)} images -> {root}")


if __name__ == "__main__":
    main()
