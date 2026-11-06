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
>   * **HRF** ([`results/hrf_xd/`](results/hrf_xd/README.md)): all models tie at matched width. At a 2× width shift,
>     which matches CHASE native, the CHASE result does **not** repeat: CAT-Net and U-Net tie on F1, and the U-Net has
>     the better AUC. The large CHASE gap appears only where the U-Net collapses, not at every large width shift.
> * **The adaptive router does not help under a shift.** On CHASE native and HRF 2× it saturates onto one branch per
>   seed and is worse than a fixed equal mix of branches.
> * **Scale-augmentation control** ([`results/scaleaug/`](results/scaleaug/README.md)): the three main models were
>   retrained with random 0.5–2× zoom, 2 seeds each.
>   * Augmentation **removes the U-Net's scale failure**: on CHASE at native scale its F1 goes from 0.506 to 0.758.
>   * The same holds on HRF at 2× (+0.06 F1 for every model).
>   * **With augmentation the CAT-Net advantage disappears.** The augmented U-Net beats both augmented CAT-Nets on
>     CHASE at every scale (F1 +0.004 to +0.007, seed-consistent) and ties them on HRF.
>   * So the CAT-Nets' cross-dataset robustness was a stand-in for data augmentation, not an architectural gain on top
>     of it.
>   * Cost on DRIVE: F1 −0.002, clDice −0.011 to −0.017.
>   * **Overall:** at this budget, no CAT-Net variant beats a U-Net trained with scale augmentation, on DRIVE or on
>     any of the three unseen datasets.
> * **Still to do:** HRF at scale 0.5 and the single-seed variants at 0.7 (running), then full-schedule GPU runs.

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
| HRF @0.35 (no TTA) | 0.99× | 0.758 / 0.836 | 0.757 / 0.835 | **0.759** / **0.837** |
| HRF @0.7 (no TTA) | 1.97× | 0.706 / 0.738 | **0.707** / **0.742** | 0.695 / 0.731 |

* **Near the training scale the models are tied**, and the U-Net has a small, seed-stable AUC lead on CHASE
  (+0.001).
* **A large CAT-Net advantage appears only on CHASE.** At CHASE native scale the fixed-mix CAT-Net beats the U-Net
  on all 28 images (F1 +0.103, AUC +0.015, clDice +0.128), and the sign holds for every seed pair. Part of the gap is
  calibration: at the oracle threshold the U-Net recovers to F1 0.61–0.65 against 0.66–0.68.
* **HRF at the same width shift (2×) shows no such gap.** F1 is +0.001, not significant, and the U-Net has the better
  AUC (+0.006, seed-consistent). There the U-Net loses only 0.05 F1 to the shift, against 0.26 on CHASE. The CAT-Net
  advantage tracks how badly the U-Net fails, not the width shift as such.
* **Adaptive routing is worse than a fixed mix under a large shift:** `ac-full` − fixed mix at CHASE 1.0 is
  F1 −0.032 and AUC −0.015, on 0 of 28 images. On HRF at 0.7 it is F1 −0.012 (1 of 45 images). The router
  saturates onto a single branch, the same branch per seed on both datasets. Under small shifts (STARE, CHASE 0.55,
  HRF 0.35) it stays mixed but brings no accuracy gain.
* **STARE is mostly a label-style shift.** Every model over-segments relative to Hoover's labels (oracle threshold
  about 0.9). The models agree with Hoover better than the second observer does: Kouznetsova scores F1 0.740.
* **Seed noise matters under a shift:** U-Net seeds span 0.467–0.539 F1 at CHASE native scale.

HRF has 45 images at 3504×2336 px. It was evaluated without flip TTA because of the image size, and is tested at 0.35
(matched) and 0.7 (2× shift). Native scale (2.8×) was not run.

Full write-ups are in [`results/chase_xd/`](results/chase_xd/README.md),
[`results/stare_xd/`](results/stare_xd/README.md) and [`results/hrf_xd/`](results/hrf_xd/README.md).

### Scale-augmentation control

Does the U-Net close the scale gap if it simply sees vessels of many widths in training? The three main models were
retrained with **random zoom, log-uniform in 0.5–2×**, on each training patch (`--scale-aug-min 0.5
--scale-aug-max 2.0`). Everything else was unchanged, and each model has 2 seeds. Full write-up:
[`results/scaleaug/`](results/scaleaug/README.md).

**F1, mean over seeds** (DRIVE and STARE: 20 images; CHASE: 28; HRF: 45, no TTA):

| Test set | U-Net | U-Net + aug | Fixed mix | Fixed mix + aug | `ac-full` | `ac-full` + aug |
|---|---|---|---|---|---|---|
| DRIVE (in-distribution) | **0.823** | 0.821 | 0.820 | 0.819 | 0.821 | 0.820 |
| CHASE @0.55 (1.05× width) | 0.769 | **0.772** | 0.764 | 0.765 | 0.766 | 0.766 |
| CHASE @0.75 (1.43×) | 0.744 | **0.776** | 0.751 | 0.770 | 0.747 | 0.769 |
| CHASE @1.0 (1.91×) | 0.506 | **0.758** | 0.610 | 0.754 | 0.578 | 0.752 |
| STARE @0.8 (1.04×) | 0.785 | **0.803** | 0.782 | 0.799 | 0.784 | 0.800 |
| STARE @1.0 (1.30×) | 0.802 | **0.807** | 0.800 | 0.806 | 0.802 | 0.805 |
| HRF @0.35 (0.99×) | 0.758 | **0.759** | 0.757 | 0.756 | **0.759** | 0.756 |
| HRF @0.7 (1.97×) | 0.706 | **0.768** | 0.707 | 0.767 | 0.695 | **0.768** |

* **Augmentation removes the scale failure.** On CHASE at native scale the augmented U-Net gains +0.251 F1 and +0.047
  AUC, on 28/28 images. Its best threshold returns from 0.05 to about 0.55. On HRF at 0.7 every model gains about
  +0.06 F1, on 45/45 images.
* **With matched training data, the U-Net is best again.** The augmented U-Net beats the augmented fixed-mix CAT-Net on
  CHASE at every scale (F1 +0.004 to +0.007, 22–27/28 images) and on STARE at 0.8 (+0.004). The sign holds for every
  seed pair. On HRF the three augmented models tie on F1 (within 0.0005), and the U-Net has the best AUC. The
  augmented CAT-Nets are statistically tied with each other.
* **Interpretation:** global and strip attention made the unaugmented CAT-Nets degrade more gracefully under an
  unseen scale. Once augmentation covers that scale, the extra context buys nothing at this budget.
* **Routing:** the augmented `ac-full` still leans 0.64–0.73 onto one branch at CHASE native scale, towards the same
  branch each seed preferred without augmentation. It is no longer worse than a fixed mix.
* **Cost on DRIVE** (flip TTA): F1 −0.001 to −0.002 and clDice −0.011 to −0.017. The augmented U-Net still has the
  best DRIVE F1.

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
