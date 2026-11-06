"""Zero-shot cross-dataset evaluation of trained checkpoints (e.g. DRIVE -> CHASE_DB1).

For one checkpoint, runs sliding-window inference (optionally with flip TTA) on every image of
the target dataset (train + test splits) at each requested scale, and saves:

* metrics at threshold 0.5 for all images and for the official test split alone;
* an *oracle* threshold (best pooled F1 on the target set). This is diagnostic only: it separates
  calibration shift (wrong operating point) from loss of discrimination (lower AUC);
* the routing report on the target images, to see whether routing adapts to the new domain.

    python scripts/cross_dataset.py --ckpt runs/drive_ac/ac-full/best.pt --name ac-full \
        --dataset chase --root data/CHASE_DB1 --scales 0.55 1.0 --tta --out results/chase_xd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from catnet.data import load_split
from catnet.evaluate import load_model
from catnet.inference import sliding_window
from catnet.metrics import dataset_metrics
from catnet.routing_analysis import routing_report


def oracle_threshold(probs, items, grid=np.linspace(0.05, 0.95, 91)):
    p = np.concatenate([pr[it["fov"] > 0.5] for pr, it in zip(probs, items)])
    y = np.concatenate([it["label"][it["fov"] > 0.5] > 0.5 for it in items])
    best = (0.5, -1.0)
    for t in grid:
        pred = p >= t
        tp = float((pred & y).sum()); fp = float((pred & ~y).sum()); fn = float((~pred & y).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="chase")
    ap.add_argument("--root", default="data/CHASE_DB1")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.55])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/chase_xd")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, _ = load_model(a.ckpt, "cpu")
    for s in a.scales:
        test_names = {it["name"] for it in load_split(a.dataset, a.root, "test", 1.0)} if s == a.scales[0] else test_names
        items = load_split(a.dataset, a.root, "train", s) + load_split(a.dataset, a.root, "test", s)
        probs = [sliding_window(model, it["image"], 256, 0.5, 8, a.tta, "cpu") for it in items]
        labels, fovs = [it["label"] for it in items], [it["fov"] for it in items]
        res = dataset_metrics(probs, labels, fovs)
        res["names"] = [it["name"] for it in items]
        idx = [i for i, it in enumerate(items) if it["name"] in test_names]
        sub = dataset_metrics([probs[i] for i in idx], [labels[i] for i in idx], [fovs[i] for i in idx])
        res["test_split_mean"] = sub["mean"]
        t, f1 = oracle_threshold(probs, items)
        res["oracle"] = {"threshold": t, "pooled_f1": f1,
                         "mean": dataset_metrics(probs, labels, fovs, threshold=t)["mean"]}
        res["routing"] = routing_report(model, items, "cpu")
        out = Path(a.out) / f"scale{s:g}" / a.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        m = res["mean"]
        print(f"[{a.name} @ {s:g}] F1 {m['f1']:.4f} AUC {m['auc']:.4f} clDice {m['cldice']:.4f} "
              f"Se {m['se']:.4f} Sp {m['sp']:.4f} | oracle t={t:.2f} F1 {res['oracle']['mean']['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
