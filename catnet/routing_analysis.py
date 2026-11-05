"""Quantify whether the context router actually adapts per pixel.

For each CAT block, on whole preprocessed images and inside the FOV only:

* ``usage``           - mean weight of each branch (an even split is 1/n each);
* ``spatial_std``     - std of each branch weight across pixels, averaged over branches
                        and images (0 means the same mix everywhere: no adaptivity);
* ``pixel_entropy``   - mean per-pixel routing entropy / log n (0 = hard choice, 1 = uniform);
* ``vessel_corr``     - Pearson correlation, per branch, between the branch weight and the
                        local vessel density (ground truth averaged to the block's resolution).
                        Non-zero values mean routing depends on the vessel content.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def routing_report(model, items, device="cpu") -> dict:
    model.eval()
    acc: dict[str, dict[str, list]] = {}
    for it in items:
        x = torch.from_numpy(it["image"])[None, None].to(device)
        model(x)
        maps = model.routing_maps() if hasattr(model, "routing_maps") else {}
        H, W = it["image"].shape
        for name, w in maps.items():
            w = w[0].float().cpu()  # (n, h, w), may include padding to a multiple of 16
            n, h, ww = w.shape
            hp, wp = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
            lab = F.pad(torch.from_numpy(it["label"])[None, None], (0, wp - W, 0, hp - H))
            fov = F.pad(torch.from_numpy(it["fov"])[None, None], (0, wp - W, 0, hp - H))
            dens = F.adaptive_avg_pool2d(lab, (h, ww))[0, 0]
            inside = F.adaptive_avg_pool2d(fov, (h, ww))[0, 0] > 0.99
            if inside.sum() < 4:
                continue
            v = w[:, inside]  # (n, P)
            d = dens[inside]
            ent = -(v * torch.log(v.clamp_min(1e-8))).sum(0).mean() / math.log(n)
            corr = []
            for i in range(n):
                a, b = v[i] - v[i].mean(), d - d.mean()
                den = float(a.norm() * b.norm())
                corr.append(float((a * b).sum()) / den if den > 1e-12 else 0.0)
            r = acc.setdefault(name, {"usage": [], "spatial_std": [], "pixel_entropy": [], "vessel_corr": []})
            r["usage"].append(v.mean(1).numpy())
            r["spatial_std"].append(float(v.std(1).mean()))
            r["pixel_entropy"].append(float(ent))
            r["vessel_corr"].append(np.array(corr))
    out = {}
    for name, r in acc.items():
        out[name] = {
            "usage": np.mean(r["usage"], 0).round(4).tolist(),
            "spatial_std": round(float(np.mean(r["spatial_std"])), 4),
            "pixel_entropy": round(float(np.mean(r["pixel_entropy"])), 4),
            "vessel_corr": np.mean(r["vessel_corr"], 0).round(4).tolist(),
        }
    return out


def summary(report: dict) -> dict[str, float]:
    """Single numbers for logging: mean spatial std and minimum branch usage over blocks."""
    if not report:
        return {}
    return {
        "route_spatial_std": float(np.mean([r["spatial_std"] for r in report.values()])),
        "route_min_usage": float(np.min([min(r["usage"]) for r in report.values()])),
        "route_pixel_entropy": float(np.mean([r["pixel_entropy"] for r in report.values()])),
    }
