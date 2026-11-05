"""Collect test_metrics.json files under a folder into a Markdown table.

Usage: python scripts/summarize.py runs/ablation/drive
"""

import json
import sys
from pathlib import Path

KEYS = ("acc", "se", "sp", "f1", "auc", "mcc", "cldice")


def main(root: str) -> None:
    rows = []
    for f in sorted(Path(root).glob("*/test_metrics.json")):
        m = json.loads(f.read_text())["mean"]
        rows.append((f.parent.name, [m.get(k, float("nan")) for k in KEYS]))
    print("| run | " + " | ".join(KEYS) + " |")
    print("|---" * (len(KEYS) + 1) + "|")
    for name, vals in rows:
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
