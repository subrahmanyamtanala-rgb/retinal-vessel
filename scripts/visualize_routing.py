"""Visualise where each CAT block routes tokens (local / global / strip).

Writes one PNG per image: preprocessed input | prediction | RGB routing map of
the first CAT block (R = local window attention, G = global grid attention,
B = directional strip mixer).

Usage:
    python scripts/visualize_routing.py --ckpt runs/drive/best.pt --dataset drive --root data/DRIVE --out vis/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from catnet.data import load_split
from catnet.evaluate import load_model
from catnet.inference import sliding_window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--block", default="enc4.0")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--out", default="vis")
    a = ap.parse_args()

    model, _ = load_model(a.ckpt, "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for it in load_split(a.dataset, a.root, a.split, a.scale, a.limit):
        img = it["image"]
        prob = sliding_window(model, img, 256) * it["fov"]
        with torch.no_grad():
            model(torch.from_numpy(img)[None, None])
        w = model.routing_maps()[a.block]
        w = F.interpolate(w, size=img.shape, mode="bilinear", align_corners=False)[0]
        rgb = torch.zeros(3, *img.shape)
        rgb[: w.shape[0]] = w  # ablated models may have < 3 branches
        rgb = (rgb.permute(1, 2, 0).numpy() * it["fov"][..., None] * 255).astype(np.uint8)
        g = lambda x: np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        Image.fromarray(np.concatenate([g(img), g(prob), rgb], 1)).save(out / f"{it['name']}_routing.png")
        print("wrote", out / f"{it['name']}_routing.png")


if __name__ == "__main__":
    main()
