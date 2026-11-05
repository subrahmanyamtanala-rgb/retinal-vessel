"""What does the router respond to? Vessels, or just position/brightness?

For every CAT block, regress each branch weight (inside the FOV, at the block's resolution) on
three per-token features and report the R^2 of each feature alone and of all three together:

* ``vessel``   - ground-truth vessel density in the token;
* ``radius``   - distance from the FOV centre, normalised to [0, 1] (pure position);
* ``intensity``- mean preprocessed intensity in the token (image appearance, no labels).

If ``radius`` explains as much as ``vessel``, the routing is mostly positional.

    python scripts/routing_confounds.py --ckpt runs/drive_ac/ac-full/best.pt --root data/DRIVE --out r.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from catnet.data import load_split
from catnet.evaluate import load_model


def r2(X: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 1e-12 else 0.0


@torch.no_grad()
def analyse(model, items, device="cpu") -> dict:
    model.eval()
    feats: dict[str, list] = {}
    for it in items:
        H, W = it["image"].shape
        model(torch.from_numpy(it["image"])[None, None].to(device))
        hp, wp = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
        pad = (0, wp - W, 0, hp - H)
        img = F.pad(torch.from_numpy(it["image"])[None, None], pad)
        lab = F.pad(torch.from_numpy(it["label"])[None, None], pad)
        fov = F.pad(torch.from_numpy(it["fov"])[None, None], pad)
        ys, xs = np.nonzero(it["fov"] > 0.5)
        cy, cx, rad = ys.mean(), xs.mean(), math.sqrt((it["fov"] > 0.5).sum() / math.pi)
        for name, w in model.routing_maps().items():
            w = w[0].float().cpu()
            n, h, ww = w.shape
            inside = F.adaptive_avg_pool2d(fov, (h, ww))[0, 0] > 0.99
            dens = F.adaptive_avg_pool2d(lab, (h, ww))[0, 0][inside]
            inten = F.adaptive_avg_pool2d(img, (h, ww))[0, 0][inside]
            gy, gx = torch.meshgrid((torch.arange(h) + 0.5) * hp / h, (torch.arange(ww) + 0.5) * wp / ww, indexing="ij")
            radius = (torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2) / rad)[inside]
            d = feats.setdefault(name, {"w": [], "vessel": [], "radius": [], "intensity": []})
            d["w"].append(w[:, inside].T.numpy())
            d["vessel"].append(dens.numpy()); d["radius"].append(radius.numpy()); d["intensity"].append(inten.numpy())
    out = {}
    for name, d in feats.items():
        Wt = np.concatenate(d["w"])
        f = {k: np.concatenate(d[k]) for k in ("vessel", "radius", "intensity")}
        allX = np.column_stack(list(f.values()))
        out[name] = {
            f"branch{i}": {**{k: round(r2(v, Wt[:, i]), 4) for k, v in f.items()},
                           "all": round(r2(allX, Wt[:, i]), 4),
                           "std": round(float(Wt[:, i].std()), 4)}
            for i in range(Wt.shape[1])
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="drive")
    ap.add_argument("--root", default="data/DRIVE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    model, _ = load_model(a.ckpt, "cpu")
    res = analyse(model, load_split(a.dataset, a.root, a.split, a.scale, None))
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
