"""Losses for thin-structure segmentation.

``VesselLoss = w_bce * BCE + w_dice * SoftDice + w_cldice * (1 - soft clDice)``

clDice (Shit et al., CVPR 2021) compares soft skeletons, so it rewards connected
vessel centre-lines rather than pixel overlap, which is dominated by the thick
vessels. All terms are evaluated inside the field-of-view mask when one is given.
Deep-supervision outputs are combined with geometrically decaying weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton of a probability map in [0, 1]."""
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_dice(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1.0) -> torch.Tensor:
    if mask is not None:
        prob, target = prob * mask, target * mask
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    return 1.0 - ((2 * inter + eps) / (prob.sum(dims) + target.sum(dims) + eps)).mean()


def soft_cldice(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                iters: int = 10, eps: float = 1.0) -> torch.Tensor:
    """Returns the clDice *score* (higher is better)."""
    if mask is not None:
        prob, target = prob * mask, target * mask
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    dims = tuple(range(1, prob.ndim))
    tprec = ((sp * target).sum(dims) + eps) / (sp.sum(dims) + eps)
    tsens = ((st * prob).sum(dims) + eps) / (st.sum(dims) + eps)
    return (2 * tprec * tsens / (tprec + tsens)).mean()


class VesselLoss(nn.Module):
    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0, w_cldice: float = 0.5,
                 pos_weight: float | None = None, skel_iters: int = 10, ds_decay: float = 0.5):
        super().__init__()
        self.w_bce, self.w_dice, self.w_cldice = w_bce, w_dice, w_cldice
        self.pos_weight = pos_weight
        self.skel_iters = skel_iters
        self.ds_decay = ds_decay

    def single(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
        bce = (bce * mask).sum() / mask.sum().clamp_min(1.0) if mask is not None else bce.mean()
        prob = torch.sigmoid(logits)
        terms = {"bce": bce, "dice": soft_dice(prob, target, mask)}
        if self.w_cldice > 0:
            terms["cldice"] = 1.0 - soft_cldice(prob, target, mask, self.skel_iters)
        terms["total"] = (self.w_bce * terms["bce"] + self.w_dice * terms["dice"]
                          + self.w_cldice * terms.get("cldice", torch.zeros((), device=logits.device)))
        return terms

    def forward(self, outputs, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        main = self.single(outs[0], target, mask)
        if len(outs) == 1:
            return main
        total, wsum = main["total"], 1.0
        for i, o in enumerate(outs[1:], start=1):
            w = self.ds_decay ** i
            total = total + w * self.single(o, target, mask)["total"]
            wsum += w
        main["total"] = total / wsum
        return main
