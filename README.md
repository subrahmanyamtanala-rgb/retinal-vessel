# CAT-Net: Context-Adaptive Transformers for Retinal Vessel Segmentation

A PyTorch implementation of **CAT-Net**, a hybrid CNN/Transformer U-shaped network for segmenting blood vessels in
colour fundus photographs. Its core component is the **Context-Adaptive Transformer (CAT) block**, which lets each
pixel choose how much local, global and directional context it uses.

> **Status:** the code is complete and tested. All experiments so far use a reduced CPU budget (24 short epochs,
> about 3 % of the full schedule).
>
> * **DRIVE** ([`results/drive_cpu/`](results/drive_cpu/README.md)): CAT-Net-S does **not** beat a size-matched U-Net
>   (F1 0.821 vs 0.823). This holds for every seed (3 U-Net seeds, 2 per main CAT-Net variant).
> * **Router fix** ([`results/drive_ac/`](results/drive_ac/README.md)): a balance loss, sharpening, temperature
>   annealing and branch dropout stop the router collapsing onto one branch. Routing becomes per-pixel and follows
>   the vessels. Segmentation is unchanged.
> * **CAT block at 1/4** ([`results/drive_q/`](results/drive_q/README.md)): routing traces the vessel tree
>   (r = 0.70 with vessel density) and clDice rises by +0.003, but F1 and AUC do not.
> * **Zero-shot cross-dataset tests** (train on DRIVE, no target images used):
>   * **CHASE_DB1** ([`results/chase_xd/`](results/chase_xd/README.md)): with vessel width matched, all models tie.
>     At native resolution (vessels 1.9× wider than in training), every CAT-Net beats every U-Net on all 28 images:
>     F1 +0.10, clDice +0.13.
>   * **STARE** ([`results/stare_xd/`](results/stare_xd/README.md)): only a 1.3× width shift. All models tie on
>     F1 and AUC, and CAT-Nets have a small, seed-consistent clDice edge (+0.002 to +0.005).
>   * **HRF** ([`results/hrf_xd/`](results/hrf_xd)): partial results only, still running.
> * **The adaptive router does not help under a shift.** On CHASE at native scale it saturates onto one branch (a
>   different one per seed), and a fixed equal mix of branches is the most robust variant. The robustness comes from
>   the transformer context, not from routing.
> * **Scale-augmentation control** ([`results/scaleaug/`](results/scaleaug)): all three main models were retrained
>   with random 0.5–2× zoom, 2 seeds each. On DRIVE this costs about 0.002 F1 and 0.012–0.017 clDice. The
>   cross-dataset tests of these models, which show whether augmentation closes the U-Net's scale gap, are running.
> * **Still to do:** finish HRF and the scale-augmentation cross-dataset tests, then full-schedule GPU runs.

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
| `stare` | `STARE/stare-images/im*.ppm` + `STARE/labels-ah/im*.ah.ppm` (`labels-vk` = 2nd observer) | first 10 / last 10 |
| `hrf` | `HRF/{images,manual1,mask}` | first 5 per category / rest |
| `folder` | `<root>/{train,test}/{images,labels,masks}/<same name>` | as given |
| `synthetic` | generated on the fly | 12 / 4 |

CHASE_DB1 and STARE have no FOV masks, so the code estimates them from the red channel (threshold 0.08; 0.18 for
STARE, whose dark border is noisier).

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
# zero-shot evaluation on all images of a dataset at several scales, with oracle threshold and routing report
python scripts/cross_dataset.py --ckpt runs/drive_catnet/best.pt --name catnet \
    --dataset chase --root data/CHASE_DB1 --scales 0.55 0.75 1.0 --tta --out results/chase_xd
```

```bash
# seed-grouped summary with paired Wilcoxon tests (works for results/{chase,stare,hrf}_xd)
python scripts/chase_summary.py --root results/chase_xd
```

```bash
# train with random-zoom scale augmentation (0.5-2x)
python -m catnet.train --config configs/drive.json --scale-aug-min 0.5 --scale-aug-max 2.0
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
| `catnet-s-q` | 3.1 M | `catnet-s-ac` + 1 CAT block at 1/4 resolution (`quarter_depth`) |
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
| same, 2nd seed | 0.9531 | 0.8468 | 0.9690 | 0.8206 | 0.9790 | 0.8429 |
| + CAT block at 1/4 (`catnet-s-q`) | 0.9529 | 0.8468 | 0.9687 | 0.8197 | 0.9789 | 0.8435 |

* **The U-Net wins** on 18 of 20 images by F1.
* **Without regularisation, the router collapses:** about 99 % of the weight goes to local attention at 1/8 and to
  global attention at 1/16, the same in every image.
* **With the anti-collapse settings, routing becomes per-pixel.** Spatial std rises from 0.007 to 0.46, all three
  branches are used, and the choices correlate with vessel density (|r| up to 0.44), not with position. F1 does not
  change (p = 0.31 vs the collapsed model).
* **A CAT block at 1/4 resolution** gives the cleanest routing: local attention follows the vessel tree, with
  R² = 0.48 against vessel density. Compared with a fixed equal mix of branches, it improves clDice on 18 of 20 images,
  but F1 and AUC do not improve.
* **Seed noise** is about 0.0004 F1 on the mean, comparable to most differences between CAT-Net variants.

The full analyses are in [`results/drive_cpu/`](results/drive_cpu/README.md),
[`results/drive_ac/`](results/drive_ac/README.md) and [`results/drive_q/`](results/drive_q/README.md).

### Seeds (DRIVE)

Extra seeds of the main models, same budget, flip TTA:

| Model | Seeds | F1 | AUC | clDice |
|---|---|---|---|---|
| U-Net | 42 / 7 / 123 | 0.8227 / 0.8224 / 0.8233 | 0.9801 / 0.9799 / 0.9801 | 0.8458 / 0.8450 / 0.8456 |
| CAT-Net-S, no router | 42 / 7 | 0.8215 / 0.8193 | 0.9790 / 0.9788 | 0.8409 / 0.8428 |
| CAT-Net-S + anti-collapse routing | 42 / 7 | 0.8210 / 0.8206 | 0.9792 / 0.9790 | 0.8418 / 0.8429 |

The U-Net is best on every seed. Runs are in [`results/seeds/`](results/seeds).

### Cross-dataset generalisation (zero-shot)

Each DRIVE checkpoint above was tested, without fine-tuning or threshold tuning, on every image of each target
dataset:

* flip TTA, threshold 0.5, metrics inside the FOV;
* paired Wilcoxon tests on seed-averaged per-image scores;
* a check that the sign of each difference holds for every pairing of seeds.

Images are rescaled so that the mean vessel width (vessel area ÷ skeleton length) matches DRIVE's 3.30 px, and are also
tested at native resolution.

| Dataset | Images | Vessel width | Matched scale |
|---|---|---|---|
| CHASE_DB1 | 28 | 6.30 px | 0.55 |
| STARE | 20 | 4.29 px | 0.8 |
| HRF | 45 | 9.30 px | 0.35 |

**F1 / clDice, mean over seeds** (U-Net: 3 seeds; fixed-mix and `ac-full`: 2 seeds each):

| Dataset @ scale | Width vs DRIVE | U-Net | CAT-Net-S, fixed mix | CAT-Net-S `ac-full` |
|---|---|---|---|---|
| CHASE @0.55 | 1.05× | **0.769** / 0.842 | 0.764 / 0.840 | 0.766 / **0.843** |
| CHASE @0.75 | 1.43× | 0.744 / 0.773 | **0.751** / **0.787** | 0.747 / 0.785 |
| CHASE @1.0 | 1.91× | 0.506 / 0.478 | **0.610** / **0.605** | 0.578 / 0.577 |
| STARE @0.8 | 1.04× | **0.785** / 0.868 | 0.782 / 0.870 | 0.784 / **0.871** |
| STARE @1.0 | 1.30× | **0.802** / 0.871 | 0.800 / 0.872 | **0.802** / **0.874** |

* **Near the training scale the models are tied**, and the U-Net has a small, seed-stable AUC lead on CHASE
  (+0.001).
* **The CAT-Net advantage grows with the vessel-width shift.** At CHASE native scale the fixed-mix CAT-Net beats the
  U-Net on all 28 images (F1 +0.103, AUC +0.015, clDice +0.128), and the sign holds for every seed pair. Part of the
  gap is calibration: at the oracle threshold the U-Net recovers to F1 0.61–0.65 against 0.66–0.68.
* **Adaptive routing is worse than a fixed mix under a large shift:** `ac-full` − fixed mix at CHASE 1.0 is
  F1 −0.032 and AUC −0.015, on 0 of 28 images. The router saturates onto a single branch, a different one for each
  seed. Under small shifts (STARE, CHASE 0.55) it stays mixed but brings no accuracy gain.
* **STARE is mostly a label-style shift.** Every model over-segments relative to Hoover's labels (oracle threshold
  about 0.9). The models agree with Hoover better than the second observer does: Kouznetsova scores F1 0.740.
* **Seed noise matters under a shift:** U-Net seeds span 0.467–0.539 F1 at CHASE native scale.

**HRF** (3504×2336 images; evaluated without TTA because of their size) is still running. At the matched scale 0.35,
the first runs tie (F1 / AUC):

| Run | F1 | AUC |
|---|---|---|
| U-Net | 0.759 | 0.983 |
| U-Net seed 123 | 0.758 | 0.983 |
| CAT-Net-S, fixed mix | 0.762 | 0.981 |
| `catnet-s-q`, adaptive | 0.757 | 0.983 |

Scales 0.5 and 0.7 (a 1.3× and 2× width shift) will show whether the CHASE pattern repeats on a third dataset.

Full write-ups are in [`results/chase_xd/`](results/chase_xd/README.md) and
[`results/stare_xd/`](results/stare_xd/README.md).

### Scale-augmentation control

Does the U-Net close the scale gap if it simply sees vessels of many widths in training? The three main models were
retrained with **random zoom, log-uniform in 0.5–2×**, on each training patch (`--scale-aug-min 0.5
--scale-aug-max 2.0`). Everything else was unchanged, and each model has 2 seeds.

DRIVE test set, F1 / clDice (no TTA; the baselines are the matching no-TTA evaluations):

| Model | Without augmentation | With 0.5–2× zoom | Δ (seed means) |
|---|---|---|---|
| U-Net (3 / 2 seeds) | 0.8213 / 0.8441 | 0.8192 / 0.8267 | −0.002 / −0.017 |
| CAT-Net-S, fixed mix (2 / 2) | 0.8190 / 0.8406 | 0.8170 / 0.8282 | −0.002 / −0.012 |
| CAT-Net-S `ac-full` (2 / 2) | 0.8191 / 0.8411 | 0.8179 / 0.8292 | −0.001 / −0.012 |

* **On DRIVE, augmentation costs a little F1 and noticeably more clDice**, and the U-Net still leads on F1.
* **The CHASE, STARE and HRF tests of the augmented models are running.** They will show whether augmentation
  removes the U-Net's native-scale failure, and whether it keeps the router from saturating.

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

The 40 tests cover:

* shape and locality of the window and grid attention;
* that router weights form a distribution;
* the anti-collapse losses, temperature and branch dropout;
* forward and backward passes for every preset;
* the AUC against a brute-force reference;
* clDice and skeleton behaviour;
* DRIVE layout discovery;
* the random-zoom scale augmentation;
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
results/                write-ups: DRIVE runs, seeds, cross-dataset (CHASE, STARE, HRF), scale augmentation
tests/
```

## License

MIT, see [LICENSE](LICENSE).
