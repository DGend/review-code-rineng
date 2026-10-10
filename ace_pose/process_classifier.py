"""Y — Process-conditional classifier (spatial GCN on major-25 skeleton).

A lightweight Graph Convolutional Network that takes a single-frame
major-25 pose and predicts the work process (one of ~16 industrial /
eldercare classes that have enough samples in our batch).

Why a GCN here?
---------------
* Each pose is a graph with 25 nodes (joints) and a fixed adjacency
  matrix (skeleton edges). GCN respects this structure better than a
  flat MLP that treats joint coordinates as an unordered vector.
* No temporal information available (we score single frames), so we
  drop the temporal dimension that "ST-GCN" usually adds and use a
  pure spatial GCN.

Inputs are normalised per-pose:
    x_norm = (x - pelvis) / ref_length

so the classifier is invariant to camera distance and worker size.

The output process probabilities are fed back into the ergonomic
analysis pipeline as a `pred_process_id` column, which lets the
`ace_pose.analysis.ErgoAnalyzer` produce per-(location × predicted-
process) summaries with no additional code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Skeleton graph (matches `ace_pose.major_joints` order)
# ----------------------------------------------------------------------

# Each edge is (from, to); we add a self-loop for every node when we
# build the adjacency matrix.
SKELETON_EDGES_25: list[tuple[int, int]] = [
    # face
    (0, 1), (0, 2),                # NOSE → EYE_L, EYE_R
    (1, 3), (2, 4),                # EYE_L → EAR_L, EYE_R → EAR_R
    (0, 23),                        # NOSE → NECK
    # upper body
    (23, 5), (23, 6),               # NECK → SH_L, SH_R
    (23, 24),                       # NECK → PELVIS
    (5, 7), (6, 8),                 # SH → EL
    (7, 9), (8, 10),                # EL → WR
    # lower body
    (24, 11), (24, 12),             # PELVIS → HIP_L, HIP_R
    (11, 13), (12, 14),             # HIP → KNEE
    (13, 15), (14, 16),             # KNEE → ANKLE
    # feet
    (15, 17), (15, 18), (15, 19),   # ANKLE_L → BIG/SMALL/HEEL_L
    (16, 20), (16, 21), (16, 22),
]

N_NODES = 25
N_FEATURES_PER_NODE = 4   # (x, y, z, confidence)


def build_normalized_adjacency() -> torch.Tensor:
    """Build the symmetric normalised adjacency Â = D^{-1/2} (A + I) D^{-1/2}.

    Returns a torch.FloatTensor of shape (N, N).
    """
    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    for s, t in SKELETON_EDGES_25:
        A[s, t] = 1.0
        A[t, s] = 1.0
    A = A + np.eye(N_NODES, dtype=np.float32)
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.from_numpy(A_hat)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Standard spectral GCN layer: H' = σ(Â H W)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in); A_hat: (N, N)
        h = torch.einsum("ij,bjk->bik", A_hat, x)
        h = self.lin(h)
        h = F.relu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class PoseGCN(nn.Module):
    """3-layer GCN + mean-pool + linear classifier."""

    def __init__(
        self,
        n_classes: int,
        in_dim: int = N_FEATURES_PER_NODE,
        hidden_dims: tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.gc1 = GCNLayer(in_dim, h1, dropout=dropout)
        self.gc2 = GCNLayer(h1, h2, dropout=dropout)
        self.gc3 = GCNLayer(h2, h3, dropout=dropout)
        self.fc = nn.Linear(h3, n_classes)
        self.register_buffer("A_hat", build_normalized_adjacency())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        h = self.gc1(x, self.A_hat)
        h = self.gc2(h, self.A_hat)
        h = self.gc3(h, self.A_hat)
        h = h.mean(dim=1)            # global mean pool over nodes
        return self.fc(h)


# ----------------------------------------------------------------------
# Pose normalisation
# ----------------------------------------------------------------------

PELVIS_IDX = 24
SHOULDER_L_IDX = 5
SHOULDER_R_IDX = 6
HIP_L_IDX = 11
HIP_R_IDX = 12
NECK_IDX = 23


def normalize_pose(kpts: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """Pelvis-centred, scale-normalised features.

    Args:
        kpts: (N=25, 3) keypoint positions.
        conf: optional (N,) per-joint confidence.

    Returns:
        (N, 4) feature tensor; last column is confidence (1.0 if not given).
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.shape != (N_NODES, 3):
        raise ValueError(f"expected (25, 3), got {kpts.shape}")
    pelvis = kpts[PELVIS_IDX]
    centred = kpts - pelvis
    sh_w = float(np.linalg.norm(kpts[SHOULDER_R_IDX] - kpts[SHOULDER_L_IDX]))
    hi_w = float(np.linalg.norm(kpts[HIP_R_IDX] - kpts[HIP_L_IDX]))
    torso = float(np.linalg.norm(kpts[NECK_IDX] - kpts[PELVIS_IDX]))
    ref = max(sh_w, hi_w, torso, 1e-6)
    centred /= ref
    c = (np.asarray(conf, dtype=np.float32)
         if conf is not None else np.ones(N_NODES, dtype=np.float32))
    return np.concatenate([centred, c[:, None]], axis=1)


# ----------------------------------------------------------------------
# Dataset / training utilities
# ----------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_class_n: int = 20
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0


class ProcessDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # features: (M, N=25, F=4); labels: (M,)
        self.x = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_features_and_labels(
    detail_dir: Path,
    summary_df,
    min_class_n: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], list[int]]:
    """Load .npz detail files into a feature matrix + labels.

    Args:
        detail_dir: directory of per-image npz blobs (`kpts_m25`, `conf_m25`).
        summary_df: pandas DataFrame from `batch_v1/summary.csv`.

    Returns:
        features: (M, 25, 4), labels: (M,) compact-encoded process indices,
        id_map: original_process_id → compact 0..K-1 index,
        used_process_ids: list of original process IDs ordered by compact index.
    """
    summary_df = summary_df.copy()
    summary_df = summary_df.loc[summary_df["status"] == "ok"]
    counts = summary_df["process_id"].value_counts()
    keep_ids = sorted(counts[counts >= min_class_n].index.tolist())
    id_map = {pid: i for i, pid in enumerate(keep_ids)}
    summary_df = summary_df.loc[summary_df["process_id"].isin(keep_ids)]

    feats = np.zeros((len(summary_df), N_NODES, N_FEATURES_PER_NODE), dtype=np.float32)
    labels = np.zeros(len(summary_df), dtype=np.int64)
    miss = 0
    rows = []
    for j, row in enumerate(summary_df.itertuples()):
        npz_path = detail_dir / f"{row.file_stem}.npz"
        if not npz_path.exists():
            miss += 1
            continue
        data = np.load(npz_path)
        kpts = data["kpts_m25"]
        conf = data["conf_m25"]
        feats[j] = normalize_pose(kpts, conf)
        labels[j] = id_map[row.process_id]
        rows.append(j)
    if not rows:
        raise RuntimeError("no detail files matched the summary")
    feats = feats[rows]
    labels = labels[rows]
    return feats, labels, id_map, keep_ids


def stratified_split(
    labels: np.ndarray,
    val_frac: float, test_frac: float, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices so each class is represented proportionally in
    train / val / test."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return (np.array(train_idx, dtype=int),
            np.array(val_idx, dtype=int),
            np.array(test_idx, dtype=int))


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------

def train_model(
    features: np.ndarray, labels: np.ndarray,
    n_classes: int, cfg: TrainConfig,
    log_fn=print,
) -> tuple[PoseGCN, dict]:
    """Train `PoseGCN` on the supplied features and labels."""
    torch.manual_seed(cfg.seed)
    train_idx, val_idx, test_idx = stratified_split(
        labels, cfg.val_frac, cfg.test_frac, seed=cfg.seed,
    )
    log_fn(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = ProcessDataset(features[train_idx], labels[train_idx])
    val_ds = ProcessDataset(features[val_idx], labels[val_idx])
    test_ds = ProcessDataset(features[test_idx], labels[test_idx])

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)

    # Class-balanced weights for imbalanced data
    class_counts = np.bincount(labels[train_idx], minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(
        1.0 / np.maximum(class_counts, 1.0), dtype=torch.float32,
    ).to(cfg.device)

    model = PoseGCN(n_classes, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y in train_dl:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss) * x.size(0)
            n_seen += x.size(0)
        train_loss = running / max(n_seen, 1)
        val_metrics = _evaluate(model, val_dl, loss_fn, cfg.device, n_classes)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        log_fn(
            f"epoch {epoch + 1:02d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.3f}  val_f1={val_metrics['macro_f1']:.3f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_dl, loss_fn, cfg.device, n_classes)
    log_fn(
        f"TEST loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.3f}  "
        f"macro_f1={test_metrics['macro_f1']:.3f}"
    )

    return model, {
        "history": history,
        "test": test_metrics,
        "splits": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }


def _evaluate(model, dl, loss_fn, device, n_classes) -> dict:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_y = []
    all_pred = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss) * x.size(0)
            n_seen += x.size(0)
            pred = logits.argmax(dim=-1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = _macro_f1(y_true, y_pred, n_classes) if y_true.size else 0.0
    return {"loss": total_loss / max(n_seen, 1), "acc": acc, "macro_f1": macro_f1,
            "y_true": y_true, "y_pred": y_pred}


def _macro_f1(y_true, y_pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ----------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------

def predict_processes(
    model: PoseGCN, features: np.ndarray, device: str | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (top-1 predicted index, full softmax prob array)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n = features.shape[0]
    all_logits = np.zeros((n, model.fc.out_features), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = torch.from_numpy(features[i:i + batch_size]).to(device)
            all_logits[i:i + batch_size] = model(chunk).cpu().numpy()
    probs = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return preds, probs
