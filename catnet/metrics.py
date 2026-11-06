"""Evaluation metrics for retinal vessel segmentation (NumPy only).

All metrics are computed on pixels inside the field-of-view (FOV) mask, the
convention used by most DRIVE / STARE / CHASE_DB1 papers. Per-image results are
aggregated by averaging; ``dataset_metrics`` also returns a pooled AUC computed over
all FOV pixels of all images.
"""

from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: P(s_pos > s_neg) + 0.5 * P(s_pos == s_neg) (Mann-Whitney U, ties averaged).

    Counts come from sorting the negatives and binary-searching each positive, which needs far less memory
    than ranking all scores together (this matters for the pooled AUC over whole HRF sets).
    """
    scores = np.asarray(scores).ravel()
    if scores.dtype not in (np.float32, np.float64):
        scores = scores.astype(np.float64)
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    neg = np.sort(scores[~labels])
    pos = np.sort(scores[labels])
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    u = int(below.sum(dtype=np.int64)) + 0.5 * int(ties.sum(dtype=np.int64))
    return float(u / (n_pos * n_neg))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, step-wise)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _skeletonize(x: np.ndarray) -> np.ndarray | None:
    try:
        from skimage.morphology import skeletonize
    except Exception:  # pragma: no cover - optional dependency
        return None
    return skeletonize(x.astype(bool))


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard clDice between binary masks (needs scikit-image; NaN otherwise)."""
    sp, sg = _skeletonize(pred), _skeletonize(gt)
    if sp is None or sg is None:
        return float("nan")
    tprec = (sp & gt.astype(bool)).sum() / max(sp.sum(), 1)
    tsens = (sg & pred.astype(bool)).sum() / max(sg.sum(), 1)
    return float(2 * tprec * tsens / max(tprec + tsens, 1e-12))


def binary_metrics(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray | None = None,
                   threshold: float = 0.5, with_cldice: bool = True) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    gt = np.asarray(gt) > 0.5
    fov = np.ones_like(gt, dtype=bool) if fov is None else np.asarray(fov) > 0.5
    p = prob[fov]
    y = gt[fov]
    yhat = p >= threshold
    tp = float(np.sum(yhat & y))
    tn = float(np.sum(~yhat & ~y))
    fp = float(np.sum(yhat & ~y))
    fn = float(np.sum(~yhat & y))
    eps = 1e-12
    se = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    pr = tp / (tp + fp + eps)
    out = {
        "acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "se": se,
        "sp": sp,
        "precision": pr,
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "mcc": (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps),
        "g_mean": float(np.sqrt(se * sp)),
        "auc": roc_auc(p, y),
        "ap": pr_auc(p, y),
    }
    if with_cldice:
        full_pred = (prob >= threshold) & fov
        out["cldice"] = cldice(full_pred, gt & fov)
    return {k: float(v) for k, v in out.items()}


def dataset_metrics(probs, gts, fovs=None, threshold: float = 0.5, with_cldice: bool = True,
                    pooled: bool = True) -> dict:
    """Per-image metrics, their mean/std, and (if ``pooled``) the AUC over all pixels pooled.

    The pooled AUC sorts every FOV pixel of the set at once; on large images (HRF) that dominates
    memory, so callers that only need the per-image means can switch it off.
    """
    fovs = fovs if fovs is not None else [None] * len(probs)
    per_image = [binary_metrics(p, g, f, threshold, with_cldice) for p, g, f in zip(probs, gts, fovs)]
    keys = per_image[0].keys() if per_image else []
    mean = {k: float(np.nanmean([m[k] for m in per_image])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in per_image])) for k in keys}
    if not pooled:
        return {"mean": mean, "std": std, "per_image": per_image}
    all_p = np.concatenate([np.asarray(p)[(np.asarray(f) > 0.5) if f is not None else slice(None)].ravel()
                            for p, f in zip(probs, fovs)])
    all_y = np.concatenate([(np.asarray(g) > 0.5)[(np.asarray(f) > 0.5) if f is not None else slice(None)].ravel()
                            for g, f in zip(gts, fovs)])
    return {"mean": mean, "std": std, "pooled_auc": roc_auc(all_p, all_y), "per_image": per_image}


def best_threshold(probs, gts, fovs=None, metric: str = "f1", grid=None) -> tuple[float, float]:
    """Pick the threshold maximising ``metric`` (use on a validation split only)."""
    grid = np.linspace(0.2, 0.8, 25) if grid is None else grid
    best = (0.5, -np.inf)
    for t in grid:
        v = dataset_metrics(probs, gts, fovs, float(t), with_cldice=False)["mean"][metric]
        if v > best[1]:
            best = (float(t), v)
    return best
