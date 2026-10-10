"""Tests for `ace_pose.process_classifier` (Y - ST-GCN-style classifier)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace_pose.process_classifier import (
    N_FEATURES_PER_NODE, N_NODES, PoseGCN, ProcessDataset,
    TrainConfig, build_normalized_adjacency, normalize_pose,
    predict_processes, stratified_split, train_model,
)


class TestAdjacency:
    def test_shape_and_symmetry(self):
        A = build_normalized_adjacency().numpy()
        assert A.shape == (N_NODES, N_NODES)
        # symmetric
        assert np.allclose(A, A.T, atol=1e-6)
        # diagonal positive (self-loop)
        assert (np.diag(A) > 0).all()


class TestNormalize:
    def test_centered_at_pelvis(self):
        kp = np.random.RandomState(0).randn(25, 3).astype(np.float32)
        f = normalize_pose(kp)
        # row 24 (pelvis) → origin after centering
        assert np.allclose(f[24, :3], 0.0, atol=1e-5)

    def test_confidence_attached(self):
        kp = np.random.RandomState(0).randn(25, 3).astype(np.float32)
        conf = np.linspace(0.1, 0.9, 25).astype(np.float32)
        f = normalize_pose(kp, conf)
        np.testing.assert_allclose(f[:, 3], conf)

    def test_scale_invariance(self):
        kp = np.random.RandomState(0).randn(25, 3).astype(np.float32) * 3.0
        f_small = normalize_pose(kp)
        f_big = normalize_pose(kp * 5.0)
        # within numerical precision, scale invariant
        np.testing.assert_allclose(f_small, f_big, atol=1e-5)


class TestModel:
    def test_forward_shape(self):
        model = PoseGCN(n_classes=5)
        x = torch.randn(4, N_NODES, N_FEATURES_PER_NODE)
        y = model(x)
        assert y.shape == (4, 5)

    def test_gradients_flow(self):
        model = PoseGCN(n_classes=3)
        x = torch.randn(2, N_NODES, N_FEATURES_PER_NODE, requires_grad=False)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestStratifiedSplit:
    def test_each_class_in_each_split(self):
        labels = np.repeat(np.arange(4), 20)
        np.random.RandomState(0).shuffle(labels)
        tr, va, te = stratified_split(labels, val_frac=0.2, test_frac=0.2, seed=0)
        for cls in range(4):
            assert (labels[tr] == cls).any()
            assert (labels[va] == cls).any()
            assert (labels[te] == cls).any()


class TestTraining:
    def test_train_overfits_small_data(self):
        """Sanity: structural class signal should be learnable.

        We build poses with three distinctive RIGHT-ELBOW positions:
          class 0: elbow below shoulder (arm hanging)
          class 1: elbow at shoulder height (arm horizontal)
          class 2: elbow above shoulder (arm raised)

        The signal survives pelvis-centred + scale normalisation,
        so a healthy GCN should easily exceed chance.
        """
        rng = np.random.default_rng(0)
        n_classes = 3
        per_class = 40
        n = n_classes * per_class

        feats = np.zeros((n, N_NODES, N_FEATURES_PER_NODE), dtype=np.float32)
        labels = np.repeat(np.arange(n_classes), per_class)
        for i in range(n):
            kp = np.zeros((25, 3), dtype=np.float32)
            # standard upright skeleton scaffold
            kp[24] = [0.0, 0.5, 0.0]                # PELVIS
            kp[23] = [0.0, -0.05, 0.0]              # NECK
            kp[5] = [-0.25, -0.05, 0.0]             # SHOULDER_L
            kp[6] = [0.25, -0.05, 0.0]              # SHOULDER_R
            kp[11] = [-0.15, 0.5, 0.0]              # HIP_L
            kp[12] = [0.15, 0.5, 0.0]               # HIP_R
            # class-dependent right elbow position
            elbow_y_offset = {0: 0.30, 1: 0.0, 2: -0.30}[int(labels[i])]
            kp[8] = [0.40, -0.05 + elbow_y_offset, 0.0]  # ELBOW_R
            kp[10] = [0.55, kp[8, 1] + 0.05, 0.0]        # WRIST_R
            # mirror left side roughly (arm hanging always)
            kp[7] = [-0.40, 0.25, 0.0]
            kp[9] = [-0.55, 0.55, 0.0]
            # legs
            kp[13] = [-0.15, 0.95, 0.0]
            kp[14] = [0.15, 0.95, 0.0]
            kp[15] = [-0.15, 1.40, 0.0]
            kp[16] = [0.15, 1.40, 0.0]
            kp[17:23] = kp[15:17].mean(0)
            # face
            kp[0] = [0.0, -0.15, 0.05]
            kp[1] = [-0.05, -0.17, 0.05]
            kp[2] = [0.05, -0.17, 0.05]
            kp[3] = [-0.10, -0.15, 0.0]
            kp[4] = [0.10, -0.15, 0.0]
            # add small noise
            kp += rng.standard_normal(kp.shape).astype(np.float32) * 0.01
            feats[i] = normalize_pose(kp)
        perm = rng.permutation(n)
        feats, labels = feats[perm], labels[perm]

        cfg = TrainConfig(
            epochs=40, batch_size=16, lr=5e-3, weight_decay=0.0, dropout=0.0,
            device="cpu", min_class_n=5, val_frac=0.2, test_frac=0.2, seed=0,
        )
        model, history = train_model(feats, labels, n_classes, cfg,
                                     log_fn=lambda _msg: None)
        assert max(history["history"]["val_acc"]) > 0.6, history["history"]["val_acc"]


class TestInference:
    def test_predict_shapes(self):
        model = PoseGCN(n_classes=7).eval()
        feats = np.random.RandomState(0).randn(11, N_NODES, N_FEATURES_PER_NODE).astype(np.float32)
        preds, probs = predict_processes(model, feats, device="cpu", batch_size=4)
        assert preds.shape == (11,)
        assert probs.shape == (11, 7)
        # softmax rows sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
