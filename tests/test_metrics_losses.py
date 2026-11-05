import numpy as np
import pytest
import torch

from catnet.losses import VesselLoss, soft_cldice, soft_dice, soft_skeleton
from catnet.metrics import best_threshold, binary_metrics, dataset_metrics, pr_auc, roc_auc


def test_roc_auc_matches_bruteforce():
    rng = np.random.default_rng(0)
    s = np.round(rng.random(300), 2)  # many ties
    y = rng.random(300) < 0.3
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert roc_auc(s, y) == pytest.approx(brute, abs=1e-12)


def test_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert pr_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)


def test_binary_metrics_known_values():
    gt = np.array([[1, 1, 0, 0]], float)
    pr = np.array([[0.9, 0.2, 0.7, 0.1]])
    m = binary_metrics(pr, gt, with_cldice=False)
    assert m["se"] == pytest.approx(0.5)
    assert m["sp"] == pytest.approx(0.5)
    assert m["acc"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_fov_excludes_pixels():
    gt = np.array([[1, 0, 0, 0]], float)
    pr = np.array([[1.0, 0.0, 0.0, 1.0]])
    fov = np.array([[1, 1, 1, 0]], float)
    assert binary_metrics(pr, gt, fov, with_cldice=False)["acc"] == pytest.approx(1.0)


def test_dataset_metrics_and_threshold():
    rng = np.random.default_rng(1)
    gts = [(rng.random((32, 32)) < 0.2).astype(float) for _ in range(3)]
    probs = [np.clip(g * 0.6 + rng.random(g.shape) * 0.4, 0, 1) for g in gts]
    r = dataset_metrics(probs, gts, with_cldice=False)
    assert r["mean"]["auc"] > 0.9 and len(r["per_image"]) == 3
    t, v = best_threshold(probs, gts)
    assert 0.2 <= t <= 0.8 and v > 0.9


def test_soft_skeleton_of_line_is_thin():
    x = torch.zeros(1, 1, 32, 32)
    x[..., 14:19, 4:28] = 1.0
    sk = soft_skeleton(x, 10)
    assert sk.sum() < x.sum() / 3
    assert sk[..., 16, 10] > 0.5


def test_losses_perfect_prediction():
    t = torch.zeros(1, 1, 32, 32)
    t[..., 10:13, :] = 1
    assert float(soft_dice(t, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(soft_cldice(t, t)) == pytest.approx(1.0, abs=1e-4)
    crit = VesselLoss()
    good = crit((t * 2 - 1) * 20, t)["total"]
    bad = crit(-(t * 2 - 1) * 20, t)["total"]
    assert float(good) < 0.05 < float(bad)


def test_deep_supervision_loss():
    t = (torch.rand(2, 1, 16, 16) > 0.8).float()
    outs = [torch.randn(2, 1, 16, 16, requires_grad=True) for _ in range(4)]
    loss = VesselLoss()(outs, t, torch.ones_like(t))["total"]
    loss.backward()
    assert all(o.grad is not None for o in outs)
