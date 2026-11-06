"""Summarise cross-dataset runs (CHASE_DB1, STARE) with seed groups.

Reads results/chase_xd/scale*/<run>/metrics.json, groups runs that differ only
by seed, and writes a markdown table plus a JSON file with:
  * per-run mean F1 / AUC / clDice,
  * per-group seed mean, seed SD and min..max,
  * paired Wilcoxon tests (per image, after averaging each group over its
    seeds) of every group against the U-Net group and of each routed model
    against its fixed-mix counterpart,
  * a "seed-robust" flag: the sign of the gap holds for every cross-seed pair
    of group means.

Usage: python scripts/chase_summary.py [--root results/chase_xd | results/stare_xd]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

GROUPS = {
    "unet": ["unet", "unet-seed7", "unet-seed123"],
    "no-router": ["catnet-s-no-router", "catnet-s-no-router-seed7"],
    "catnet-s": ["catnet-s"],
    "ac-full": ["ac-full", "ac-full-seed7"],
    "q-norouter": ["q-norouter"],
    "q-full": ["q-full"],
}
COMPARE = [("no-router", "unet"), ("catnet-s", "unet"), ("ac-full", "unet"),
           ("q-norouter", "unet"), ("q-full", "unet"),
           ("ac-full", "no-router"), ("catnet-s", "no-router"),
           ("q-full", "q-norouter")]
METRICS = ["f1", "auc", "cldice"]


def load(root: Path, scale: str, run: str):
    p = root / f"scale{scale}" / run / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    per = {m: np.array([r[m] for r in d["per_image"]]) for m in METRICS}
    return {"mean": {m: float(d["mean"][m]) for m in METRICS}, "per": per,
            "names": d["names"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/chase_xd")
    args = ap.parse_args()
    root = Path(args.root)
    scales = sorted((p.name[5:] for p in root.glob("scale*")), key=float)

    out, md = {}, []
    for s in scales:
        runs = {g: {r: load(root, s, r) for r in rs} for g, rs in GROUPS.items()}
        runs = {g: {r: v for r, v in rs.items() if v is not None} for g, rs in runs.items()}
        res = {"groups": {}, "tests": {}}
        md.append(f"\n### {root.name} scale {s}\n")
        md.append("| group | seeds | F1 mean ± SD (min..max) | AUC mean ± SD | clDice mean ± SD |")
        md.append("|---|---|---|---|---|")
        avg = {}
        for g, rs in runs.items():
            if not rs:
                continue
            row = {"runs": {r: v["mean"] for r, v in rs.items()}}
            cells = []
            for m in METRICS:
                vals = np.array([v["mean"][m] for v in rs.values()])
                sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[m] = {"mean": float(vals.mean()), "sd": sd,
                          "min": float(vals.min()), "max": float(vals.max())}
                rng = f" ({vals.min():.4f}..{vals.max():.4f})" if len(vals) > 1 else ""
                cells.append(f"{vals.mean():.4f} ± {sd:.4f}{rng}" if m == "f1"
                             else f"{vals.mean():.4f} ± {sd:.4f}")
            md.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
            avg[g] = {m: np.mean([v["per"][m] for v in rs.values()], axis=0) for m in METRICS}
            res["groups"][g] = row

        md.append("\n| comparison | metric | Δ (seed-avg) | wins/n | p (Wilcoxon) | sign holds for all seed pairs |")
        md.append("|---|---|---|---|---|---|")
        for a, b in COMPARE:
            if a not in avg or b not in avg:
                continue
            for m in METRICS:
                d = avg[a][m] - avg[b][m]
                try:
                    p = float(wilcoxon(avg[a][m], avg[b][m]).pvalue)
                except ValueError:
                    p = 1.0
                pairs = [runs[a][ra]["mean"][m] - runs[b][rb]["mean"][m]
                         for ra, rb in itertools.product(runs[a], runs[b])]
                robust = all(x > 0 for x in pairs) or all(x < 0 for x in pairs)
                res["tests"][f"{a} vs {b} [{m}]"] = {
                    "delta": float(d.mean()), "wins": int((d > 0).sum()), "n": int(d.size),
                    "p": p, "pair_deltas": [float(x) for x in pairs], "seed_robust": robust}
                md.append(f"| {a} − {b} | {m} | {d.mean():+.4f} | {(d > 0).sum()}/{d.size} | "
                          f"{p:.2g} | {'yes' if robust else 'no'} "
                          f"({min(pairs):+.4f}..{max(pairs):+.4f}) |")
        out[s] = res

    (root / "seed_summary.json").write_text(json.dumps(out, indent=2))
    (root / "seed_summary.md").write_text("\n".join(md).lstrip() + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
