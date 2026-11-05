"""Train CAT-Net.

Example::

    python -m catnet.train --dataset drive --root data/DRIVE --out runs/drive
    python -m catnet.train --config configs/drive.json
    python -m catnet.train --dataset synthetic --epochs 3 --out runs/smoke   # no data needed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatchDataset, load_split, write_synthetic_dataset
from .inference import sliding_window
from .losses import VesselLoss
from .metrics import dataset_metrics
from .routing_analysis import routing_report, summary as routing_summary
from .models import MODEL_NAMES, build_model, count_parameters

DEFAULTS = dict(
    dataset="drive", root="data/DRIVE", out="runs/catnet", model="catnet", scale=1.0,
    patch=128, batch_size=16, samples_per_epoch=4000, epochs=200, lr=1e-3, weight_decay=0.05,
    warmup_epochs=5, w_bce=1.0, w_dice=1.0, w_cldice=0.5, val_fraction=0.1, val_every=5,
    eval_patch=256, seed=42, workers=2, threads=0, amp=True, device="auto", limit=None, model_kw={},
    # anti-collapse routing (all off by default = original behaviour)
    router_balance=0.0, router_entropy=0.0, router_tau_start=0.0, router_tau_end=1.0,
)


def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, help="JSON file with any of the options below")
    for k, v in DEFAULTS.items():
        if k == "model_kw":
            ap.add_argument("--model-kw", type=json.loads, default=None, help="JSON dict of CATNetConfig overrides")
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            typ = type(v) if v is not None else int
            ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=None,
                            choices=MODEL_NAMES if k == "model" else None)
    a = vars(ap.parse_args(argv))
    cfg = dict(DEFAULTS)
    cfg_path = a.pop("config")
    if cfg_path:
        cfg.update(json.loads(Path(cfg_path).read_text()))
    cfg.update({k: v for k, v in a.items() if v is not None})
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(d: str) -> torch.device:
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def prepare_data(cfg: dict):
    if cfg["dataset"] == "synthetic":
        root = Path(tempfile.mkdtemp(prefix="catnet_syn_"))
        write_synthetic_dataset(root, n_train=12, n_test=4, size=192, seed=cfg["seed"])
        ds, root_ = "folder", root
    else:
        ds, root_ = cfg["dataset"], cfg["root"]
    train = load_split(ds, root_, "train", cfg["scale"], cfg["limit"])
    test = load_split(ds, root_, "test", cfg["scale"], cfg["limit"])
    rng = np.random.default_rng(cfg["seed"])
    n_val = max(1, int(round(len(train) * cfg["val_fraction"]))) if len(train) > 1 and cfg["val_fraction"] > 0 else 0
    idx = rng.permutation(len(train))
    val = [train[i] for i in idx[:n_val]]
    train = [train[i] for i in idx[n_val:]]
    return train, val, test


def evaluate(model, items, cfg, device, tta=False) -> dict:
    probs = [sliding_window(model, it["image"], cfg["eval_patch"], 0.5, 8, tta, device) for it in items]
    res = dataset_metrics(probs, [it["label"] for it in items], [it["fov"] for it in items])
    res["names"] = [it["name"] for it in items]
    return res, probs


def lr_lambda(cfg, steps_per_epoch):
    warm = cfg["warmup_epochs"] * steps_per_epoch
    total = cfg["epochs"] * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    return f


def main(argv=None) -> dict:
    cfg = parse_args(argv)
    seed_everything(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    device = resolve_device(cfg["device"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    train_items, val_items, test_items = prepare_data(cfg)
    print(f"[data] train={len(train_items)} val={len(val_items)} test={len(test_items)} device={device}")
    ds = PatchDataset(train_items, cfg["patch"], cfg["samples_per_epoch"], seed=cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                    drop_last=True, pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0)

    model = build_model(cfg["model"], **cfg.get("model_kw", {})).to(device)
    print(f"[model] {cfg['model']}: {count_parameters(model) / 1e6:.2f} M parameters")
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "rel_bias" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, len(dl)))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = VesselLoss(cfg["w_bce"], cfg["w_dice"], cfg["w_cldice"])

    best, history = -1.0, []
    has_router = hasattr(model, "router_loss")
    for epoch in range(cfg["epochs"]):
        ds.set_epoch(epoch)
        if has_router and cfg["router_tau_start"] > 0:
            frac = epoch / max(cfg["epochs"] - 1, 1)
            tau = cfg["router_tau_start"] + (cfg["router_tau_end"] - cfg["router_tau_start"]) * frac
            model.set_router_tau(tau)
            model.cfg.router_tau = tau  # saved with the checkpoint so evaluation uses the same tau
        model.train()
        t0 = time.time()
        run = {"total": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0, "balance": 0.0, "entropy": 0.0}
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                outs = model(x)
            outs = [o.float() for o in outs] if isinstance(outs, list) else outs.float()
            loss = crit(outs, y, m)
            rl = model.router_loss() if has_router else None
            if rl is not None:
                loss.update(rl)
                loss["total"] = loss["total"] + cfg["router_balance"] * rl["balance"] + cfg["router_entropy"] * rl["entropy"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in run:
                if k in loss:
                    run[k] += float(loss[k].detach())
        log = {"epoch": epoch + 1, "lr": sched.get_last_lr()[0], "time": time.time() - t0,
               **{f"loss_{k}": v / len(dl) for k, v in run.items()}}
        last = epoch + 1 == cfg["epochs"]
        if val_items and ((epoch + 1) % cfg["val_every"] == 0 or last):
            vres, _ = evaluate(model, val_items, cfg, device)
            log.update({f"val_{k}": v for k, v in vres["mean"].items() if k in ("f1", "auc", "se", "sp", "cldice")})
            log.update(routing_summary(routing_report(model, val_items, device)))
            score = vres["mean"]["f1"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(), "train_cfg": cfg,
                            "epoch": epoch + 1}, out / "best.pt")
        history.append(log)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in log.items()}))
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(), "train_cfg": cfg,
                "epoch": cfg["epochs"]}, out / "last.pt")
    if not (out / "best.pt").exists():
        (out / "best.pt").write_bytes((out / "last.pt").read_bytes())
    (out / "history.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    tres, _ = evaluate(model, test_items, cfg, device)
    tres["routing"] = routing_report(model, test_items, device)
    (out / "test_metrics.json").write_text(json.dumps(tres, indent=2))
    print("[test]", json.dumps({k: round(v, 4) for k, v in tres["mean"].items()}))
    return tres


if __name__ == "__main__":
    main()
