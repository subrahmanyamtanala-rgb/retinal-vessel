"""Evaluate a trained checkpoint, optionally cross-dataset, and save probability maps.

Example::

    python -m catnet.evaluate --ckpt runs/drive/best.pt --dataset drive --root data/DRIVE --tta
    python -m catnet.evaluate --ckpt runs/drive/best.pt --dataset chase --root data/CHASE_DB1 --scale 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import load_split
from .inference import sliding_window
from .metrics import dataset_metrics
from .models import CATNet, CATNetConfig


def load_model(ckpt_path: str, device) -> tuple[CATNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CATNet(CATNetConfig.from_dict(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor (match the training resolution)")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default=None, help="folder for metrics.json and probability maps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    model, _ = load_model(a.ckpt, a.device)
    items = load_split(a.dataset, a.root, a.split, a.scale)
    probs = [sliding_window(model, it["image"], a.patch, a.overlap, 8, a.tta, a.device) for it in items]
    res = dataset_metrics(probs, [it["label"] for it in items], [it["fov"] for it in items], a.threshold)
    res["names"] = [it["name"] for it in items]
    print(json.dumps({k: round(v, 4) for k, v in res["mean"].items()}, indent=2))
    if a.out:
        out = Path(a.out)
        (out / "prob").mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        for it, p in zip(items, probs):
            Image.fromarray((np.clip(p * it["fov"], 0, 1) * 255).astype(np.uint8)).save(out / "prob" / f"{it['name']}.png")
    return res


if __name__ == "__main__":
    main()
