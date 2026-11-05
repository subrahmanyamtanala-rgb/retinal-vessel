"""Whole-image inference by overlapping sliding windows with Gaussian blending."""

from __future__ import annotations

import numpy as np
import torch


def _gaussian(p: int, sigma_frac: float = 0.25) -> torch.Tensor:
    ax = torch.arange(p, dtype=torch.float32) - (p - 1) / 2
    g = torch.exp(-(ax**2) / (2 * (p * sigma_frac) ** 2))
    w = g[:, None] * g[None, :]
    return (w / w.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window(model, image: np.ndarray, patch: int = 256, overlap: float = 0.5,
                   batch_size: int = 8, tta: bool = False, device: str | torch.device = "cpu") -> np.ndarray:
    """Return the vessel probability map ``(H, W)`` for a preprocessed image ``(H, W)``."""
    model.eval()
    img = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
    H, W = img.shape
    p = patch
    ph, pw = max(p - H, 0), max(p - W, 0)
    if ph or pw:
        img = torch.nn.functional.pad(img[None, None], (0, pw, 0, ph), mode="reflect")[0, 0]
    Hp, Wp = img.shape
    step = max(1, int(p * (1 - overlap)))
    ys = list(range(0, Hp - p + 1, step))
    xs = list(range(0, Wp - p + 1, step))
    if ys[-1] != Hp - p:
        ys.append(Hp - p)
    if xs[-1] != Wp - p:
        xs.append(Wp - p)
    coords = [(y, x) for y in ys for x in xs]
    wmap = _gaussian(p).to(device)
    acc = torch.zeros(Hp, Wp, device=device)
    norm = torch.zeros(Hp, Wp, device=device)
    img = img.to(device)
    for i in range(0, len(coords), batch_size):
        cs = coords[i : i + batch_size]
        x = torch.stack([img[y : y + p, x0 : x0 + p] for y, x0 in cs])[:, None]
        prob = _predict(model, x, tta)
        for (y, x0), pr in zip(cs, prob[:, 0]):
            acc[y : y + p, x0 : x0 + p] += pr * wmap
            norm[y : y + p, x0 : x0 + p] += wmap
    return (acc / norm)[:H, :W].cpu().numpy()


def _predict(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    prob = torch.sigmoid(out)
    if not tta:
        return prob
    n = 1
    for dims in ((-1,), (-2,), (-1, -2)):
        o = model(torch.flip(x, dims))
        o = o[0] if isinstance(o, (list, tuple)) else o
        prob = prob + torch.flip(torch.sigmoid(o), dims)
        n += 1
    return prob / n
