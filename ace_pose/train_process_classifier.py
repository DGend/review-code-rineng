"""Train the Y module on the current batch outputs.

Reads batch_v1/summary.csv plus the per-image .npz details, trains the
spatial GCN, evaluates on a held-out test split, and writes:
  - model weights → batch_v1/process_classifier.pt
  - predictions for ALL rows → batch_v1/summary_with_pred.csv
  - training history + metrics → batch_v1/process_training.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.process_classifier import (
    PoseGCN, TrainConfig, build_features_and_labels, predict_processes,
    train_model,
)

log = logging.getLogger("ace_pose.train_process_classifier")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_dir = Path(args.batch)
    summary_csv = batch_dir / "summary.csv"
    detail_dir = batch_dir / "details"
    df = pd.read_csv(summary_csv)

    log.info("Loaded %d rows", len(df))
    log.info("Building features from %s", detail_dir)
    feats, labels, id_map, used_ids = build_features_and_labels(
        detail_dir, df, min_class_n=20,
    )
    log.info("feats=%s labels=%s n_classes=%d",
             feats.shape, labels.shape, len(id_map))
    log.info("Process IDs used (compact -> original): %s",
             {i: pid for pid, i in id_map.items()})

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=64, lr=1e-3, weight_decay=1e-4,
        dropout=0.3, device=args.device, min_class_n=20,
        val_frac=0.15, test_frac=0.15, seed=args.seed,
    )
    model, hist = train_model(feats, labels, len(id_map), cfg, log_fn=log.info)

    # Predict process for every viable image, then merge back to the
    # full summary (rows not in `used_ids` get pred = NaN).
    preds_compact, probs = predict_processes(model, feats, device=args.device)
    inv_id_map = {i: pid for pid, i in id_map.items()}
    preds_original = np.array([inv_id_map[int(c)] for c in preds_compact])

    # Re-attach to summary by file_stem
    sub = df.loc[df["process_id"].isin(used_ids) & (df["status"] == "ok")].copy()
    sub = sub.sort_values("file_stem").reset_index(drop=True)
    # The order from build_features_and_labels follows sub's iteration order
    # but skipped missing npz files; we re-run a robust merge instead.
    npz_present = [
        (detail_dir / f"{stem}.npz").exists()
        for stem in sub["file_stem"]
    ]
    sub = sub.loc[npz_present].reset_index(drop=True)
    assert len(sub) == len(preds_compact), (len(sub), len(preds_compact))
    sub["pred_process_id"] = preds_original
    sub["pred_process_correct"] = (sub["pred_process_id"] == sub["process_id"])
    sub["pred_process_top_prob"] = probs.max(axis=1)
    sub["pred_process_entropy"] = (
        -np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0).sum(axis=1)
    )

    out = df.merge(
        sub[["file_stem", "pred_process_id", "pred_process_correct",
             "pred_process_top_prob", "pred_process_entropy"]],
        on="file_stem", how="left",
    )

    out_csv = batch_dir / "summary_with_pred.csv"
    out.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows, %d with predictions)",
             out_csv, len(out), out["pred_process_id"].notna().sum())

    # Save model weights
    weights_path = batch_dir / "process_classifier.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "id_map": id_map,
        "used_process_ids": used_ids,
        "n_classes": len(id_map),
        "config": cfg.__dict__,
    }, weights_path)
    log.info("Wrote %s", weights_path)

    # Save metrics
    metrics_path = batch_dir / "process_training.json"
    test = hist["test"]
    summary_metrics = {
        "n_classes": len(id_map),
        "used_process_ids": used_ids,
        "test_loss": test["loss"],
        "test_accuracy": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "train_loss": hist["history"]["train_loss"],
        "val_loss": hist["history"]["val_loss"],
        "val_acc": hist["history"]["val_acc"],
        "val_macro_f1": hist["history"]["val_macro_f1"],
    }
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    log.info("Wrote %s", metrics_path)

    # Per-class report
    y_true = test["y_true"]
    y_pred = test["y_pred"]
    log.info("Per-class test counts and accuracy:")
    for ci in range(len(id_map)):
        n = int((y_true == ci).sum())
        if n == 0:
            continue
        acc = float(((y_pred == ci) & (y_true == ci)).sum() / max(n, 1))
        log.info("  process_id=%d (compact %d) n=%d  recall=%.3f",
                 inv_id_map[ci], ci, n, acc)


if __name__ == "__main__":
    main()
