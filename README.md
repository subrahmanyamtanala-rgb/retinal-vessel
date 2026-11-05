# CAT-Net: Context-Adaptive Transformers for Retinal Vessel Segmentation

A PyTorch implementation of **CAT-Net**, a hybrid CNN/Transformer U-shaped network for segmenting blood vessels in
colour fundus photographs. Its core component is the **Context-Adaptive Transformer (CAT) block**, which lets each
pixel choose how much local, global and directional context it uses.

> **Status:** the code is complete and tested.
>
> * **First DRIVE run** ([`results/drive_cpu/`](results/drive_cpu/README.md)), at about 3 % of the full training
>   schedule on CPU: CAT-Net-S did **not** beat a size-matched U-Net (F1 0.821 vs 0.823), and its context router
>   collapsed to one fixed branch per stage.
> * **Router fix** ([`results/drive_ac/`](results/drive_ac/README.md)): a balance loss, sharpening, temperature
>   annealing and branch dropout fix the collapse. Routing is now per-pixel and follows the vessels, not image
>   position. Segmentation is unchanged (F1 0.821), so the gap to the U-Net remains.
> * **Still to do:** full-schedule GPU runs.

## Why context-adaptive?

Retinal vessels need different context in different places:

* **Capillaries** (1–3 px wide) need fine, local detail.
* **Gaps** where vessels cross the optic disc, pass through lesions or fade in low contrast need **long-range
  context** to keep the vessel tree connected.
* Vessels are **elongated and oriented**, which favours direction-selective filters.
* Images from different **fundus cameras** vary widely in illumination and contrast.

A standard transformer block uses one fixed attention pattern everywhere. A CAT block runs three token mixers in
parallel and blends them per pixel:

| Branch | What it does | What it is good at |
|---|---|---|
| Local window attention | MHSA inside `w×w` windows, relative-position bias | capillaries, vessel edges |
| Global grid attention | MHSA across a sparse `g×g` grid spanning the whole map | continuity across the disc, lesions and low-contrast gaps |
| Directional strip mixer | depth-wise `1×k` + `k×1` convolutions | elongated, oriented structures |

* A **context router** predicts per-pixel branch weights
  `softmax(local_head(DWConv(x)) + global_head([mean(x), std(x)]))`. Its last layers are zero-initialised, so
  training starts from an even mix.
* A **context-adaptive FFN** FiLM-modulates its hidden channels with image-level statistics so it can adapt to
  illumination and contrast shift between datasets.
* **Context-adaptive skip gates** in the decoder use the decoder feature, which already carries global context, to
  damp background texture (choroid, optic-disc rim, lesions) in the encoder skips.

```
input ─ Conv(1/1) ─ Conv(1/2) ─ Conv(1/4) ─ CAT×2 (1/8) ─ CAT×2 (1/16)
            │          │          │            │                │
            └── CAG ◄──┴── CAG ◄──┴─── CAG ◄───┴──── CAG ◄──────┘  decoder
            ▼
         logits (+ deep-supervision heads at 1/2, 1/4, 1/8)
```

The first three stages are convolutional: at full resolution, vessels are only a few pixels wide and convolutions are
cheaper and better suited. Attention starts at 1/8 resolution, where a 128 px patch becomes 16×16 tokens.

**Loss:** `BCE + Dice + 0.5·(1 − soft clDice)`, computed inside the FOV mask. clDice rewards a connected vessel
skeleton, not just pixel overlap.

## Installation

```bash
pip install -e ".[dev]"
```

On a CPU-only machine, install the CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data

Put the datasets under `data/` (or pass `--root`):

| `--dataset` | Expected layout | Split |
|---|---|---|
| `drive` | `DRIVE/{training,test}/{images,1st_manual,mask}` | official 20 / 20 |
| `chase` | `CHASE_DB1/Image_XXY.jpg` + `Image_XXY_1stHO.png` | first 20 / last 8 |
| `stare` | `STARE/stare-images/im*.ppm` + `STARE/labels-ah/im*.ah.ppm` | first 10 / last 10 |
| `hrf` | `HRF/{images,manual1,mask}` | first 5 per category / rest |
| `folder` | `<root>/{train,test}/{images,labels,masks}/<same name>` | as given |
| `synthetic` | generated on the fly | 12 / 4 |

CHASE_DB1 and STARE have no FOV masks, so the code estimates them from the red channel.

**Preprocessing:** green channel → standardise within the FOV → CLAHE → gamma 1.2.

## Usage

```bash
# quick smoke test, no data needed (~1 min on CPU)
python -m catnet.train --dataset synthetic --model catnet-s --epochs 3 --samples-per-epoch 64 --batch-size 4 --patch 96 --workers 0
```

```bash
# train on DRIVE
python -m catnet.train --config configs/drive.json
```

```bash
# evaluate with flip test-time augmentation and save probability maps
python -m catnet.evaluate --ckpt runs/drive_catnet/best.pt --dataset drive --root data/DRIVE --tta --out results/drive
```

```bash
# cross-dataset generalisation: train on DRIVE, test on CHASE_DB1 (rescale to roughly match vessel width)
python -m catnet.evaluate --ckpt runs/drive_catnet/best.pt --dataset chase --root data/CHASE_DB1 --scale 0.6
```

```bash
# see which branch the router picks at each pixel (R = local, G = global, B = strip)
python scripts/visualize_routing.py --ckpt runs/drive_catnet/best.pt --dataset drive --root data/DRIVE --out vis
```

```bash
# full ablation study
scripts/run_ablations.sh configs/drive.json
```

Training writes `config.json`, `history.json`, `best.pt` (best validation F1), `last.pt` and `test_metrics.json` to
`--out`.

### Models

| Preset | Params | Notes |
|---|---|---|
| `catnet-s` | 2.9 M | small; widths 16–192, 1+1 CAT blocks |
| `catnet` | 15.7 M | default; widths 32–320, 2+2 CAT blocks |
| `catnet-l` | 55.2 M | large; widths 48–512, 2+4 CAT blocks |
| `catnet-s-ac` | 2.9 M | `catnet-s` + branch dropout; use with the `--router-*` losses below |
| `catnet-no-router` | | equal fixed branch weights |
| `catnet-no-film` | | FFN ignores image-level context |
| `catnet-no-gate` | | plain U-Net skips |
| `unet` | | residual U-Net baseline (set widths with `--model-kw`) |
| `catnet-local-only` / `-global-only` / `-no-strip` | | branch ablations |

Any `CATNetConfig` field can be overridden, for example `--model-kw '{"window": 4, "grid": 8}'`.

Without regularisation, the router tends to collapse onto one branch per stage. The recommended anti-collapse
settings are:

```bash
python -m catnet.train --config configs/drive.json --model catnet-s-ac \
    --router-balance 0.1 --router-entropy 0.02 --router-tau-start 2 --router-tau-end 1
```

* `--router-balance` adds a load-balancing loss so that all branches are used.
* `--router-entropy` sharpens each pixel's choice of branch.
* `--router-tau-start` / `--router-tau-end` anneal the router temperature from soft to sharp over training.

## Metrics

Metrics are computed inside the FOV, per image, then averaged. They include accuracy, sensitivity, specificity,
precision, F1/Dice, IoU, MCC, G-mean, ROC-AUC, average precision and clDice (skeleton-based connectivity).
`dataset_metrics` also reports a pooled AUC over all FOV pixels, as many papers do. Pick the threshold on the
validation split (`catnet.metrics.best_threshold`), never on the test set.

## Results

### DRIVE, reduced CPU budget

These runs used 24 k training patches, about 3 % of the full schedule, with a single seed. All models had about
2.9 M parameters. Scores are means over the 20 test images, with flip TTA.

| Model | Acc | Se | Sp | F1 | AUC | clDice |
|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.9538 | 0.8464 | 0.9699 | **0.8227** | **0.9801** | **0.8458** |
| CAT-Net-S, no router | 0.9538 | 0.8395 | 0.9709 | 0.8215 | 0.9790 | 0.8409 |
| CAT-Net-S | 0.9535 | 0.8413 | 0.9702 | 0.8207 | 0.9790 | 0.8418 |
| CAT-Net-S + anti-collapse routing | 0.9535 | 0.8430 | 0.9699 | 0.8210 | 0.9792 | 0.8418 |

* **The U-Net wins** on 18 of 20 images by F1.
* **Without regularisation, the router collapses:** about 99 % of the weight goes to local attention at 1/8 and to
  global attention at 1/16, the same in every image.
* **With the anti-collapse settings, routing becomes per-pixel.** Spatial std rises from 0.007 to 0.46, all three
  branches are used, and the choices correlate with vessel density (|r| up to 0.44), not with position. F1 does not
  change (p = 0.31 vs the collapsed model).

The full analyses are in [`results/drive_cpu/`](results/drive_cpu/README.md) and
[`results/drive_ac/`](results/drive_ac/README.md).

### Full schedule

| Dataset | Acc | Se | Sp | F1 | AUC | clDice |
|---|---|---|---|---|---|---|
| DRIVE | – | – | – | – | – | – |
| CHASE_DB1 | – | – | – | – | – | – |
| STARE | – | – | – | – | – | – |

These still need GPU runs with `configs/*.json`.

## Tests

```bash
pytest -q
```

The 37 tests cover:

* shape and locality of the window and grid attention;
* that router weights form a distribution;
* the anti-collapse losses, temperature and branch dropout;
* forward and backward passes for every preset;
* the AUC against a brute-force reference;
* clDice and skeleton behaviour;
* DRIVE layout discovery;
* sliding-window inference;
* an end-to-end train → checkpoint → reload run.

## Repository layout

```
catnet/
  models/cat_block.py   CAT block: window/grid attention, strip mixer, router, adaptive FFN
  models/catnet.py      CAT-Net, skip gates, presets
  losses.py             BCE + Dice + soft clDice, deep supervision
  metrics.py            FOV-masked metrics, AUC, AP, clDice
  data/                 dataset discovery, preprocessing, patch sampler, synthetic generator
  inference.py          sliding-window inference with Gaussian blending and flip TTA
  routing_analysis.py   per-block routing statistics (usage, spatial std, entropy, vessel correlation)
  train.py, evaluate.py command-line entry points
configs/                per-dataset training configs
scripts/                ablations, result tables, routing visualisation and confound analysis
results/                write-ups of the DRIVE runs
tests/
```

## License

MIT, see [LICENSE](LICENSE).
