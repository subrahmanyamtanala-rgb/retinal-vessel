#!/usr/bin/env bash
# Train every ablation preset on one dataset with identical settings.
# Usage: scripts/run_ablations.sh configs/drive.json
set -euo pipefail
CFG=${1:-configs/drive.json}
for M in catnet catnet-no-router catnet-no-film catnet-no-gate catnet-local-only catnet-global-only catnet-no-strip; do
  python -m catnet.train --config "$CFG" --model "$M" --out "runs/ablation/$(basename "$CFG" .json)/$M"
done
python scripts/summarize.py runs/ablation/$(basename "$CFG" .json)
