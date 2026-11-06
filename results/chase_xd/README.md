# Cross-dataset: train on DRIVE, test on CHASE_DB1 (zero-shot)

**Question:** do context-adaptive transformer blocks make a vessel segmenter more robust to a dataset shift? CHASE_DB1
differs from DRIVE in camera, image size, field of view and population (children). No CHASE image was used for
training, validation or threshold selection.

**Bottom line**

* **When vessel size is matched (scale 0.55), there is no robustness benefit.** Every model lands at F1 0.762–0.770
  and AUC ≈ 0.980. The U-Net has the best AUC, by about 0.001, and this holds for every seed pairing.
* **The ranking flips under a scale shift.** At native resolution (scale 1.0) vessels are about 1.9× wider than
  anything seen in training.
  * Every CAT-Net beats every U-Net on all 28 images: F1 +0.07 to +0.10 and clDice +0.09 to +0.13, averaged over seeds.
  * The sign holds for every seed pair.
  * At 0.75 the gap is smaller but still holds for clDice: +0.012 to +0.018, for every seed pairing.
* **Adaptive routing hurts under the scale shift.**
  * `ac-full` is worse than the same network with a fixed equal branch mix on all 28 images at native scale: F1 −0.032,
    AUC −0.015.
  * Routing that is per-pixel and vessel-driven on DRIVE saturates onto a single branch on unseen-scale images. Which
    branch it saturates onto differs between seeds.
  * The robustness comes from the **transformer context itself**, not from the router.
* **Under a shift, seed noise is as large as the effect.** Two `ac-full` seeds differ by 0.042 F1 at native scale, and
  three U-Net seeds span 0.467–0.539. A single-seed comparison would have been misleading. The extra seeds are why this
  README exists in this form.

## Setup

* **Checkpoints:** the DRIVE-trained runs from [`../drive_cpu`](../drive_cpu/README.md),
  [`../drive_ac`](../drive_ac/README.md), [`../drive_q`](../drive_q/README.md) and [`../seeds`](../seeds). They all
  use the same 24-epoch CPU schedule and the same 2-image validation split.
* **Data:** all 28 CHASE_DB1 images. The first observer (`_1stHO`) is the reference. The FOV is estimated (CHASE ships
  no masks; the estimated masks were checked visually).
* **Scale:**
  * Mean vessel width (area / skeleton length) is 3.30 px on DRIVE and 6.30 px on CHASE.
  * FOV diameter is 538 vs 918 px.
  * So the matched scale is 0.52 (from vessel width) to 0.59 (from FOV diameter). We use **0.55** as matched, **0.75**
    as a partial mismatch and **1.0** as native.
* **Inference:** sliding window with flip TTA, threshold 0.5, metrics inside the FOV.
* **Statistics:** for groups with several seeds, each image's metric is averaged over seeds before a paired Wilcoxon
  test across the 28 images. *Sign holds for all seed pairs* means the difference of means keeps its sign for every
  combination of one seed from each group.
* **Oracle threshold:** the best pooled F1 over a grid from 0.05 to 0.95. It is a diagnostic only, used to separate
  calibration shift from loss of discrimination.

| Group | Runs (seeds) | Model |
|---|---|---|
| `unet` | `unet`, `unet-seed7`, `unet-seed123` | U-Net baseline, 3.1 M |
| `no-router` | `catnet-s-no-router`, `-seed7` | CAT-Net-S, fixed equal branch mix |
| `catnet-s` | `catnet-s` | CAT-Net-S, adaptive routing without the anti-collapse recipe |
| `ac-full` | `ac-full`, `ac-full-seed7` | CAT-Net-S, adaptive routing + anti-collapse recipe |
| `q-norouter` / `q-full` | one seed each | `catnet-s-q` (extra CAT block at 1/4), fixed vs adaptive |

**Reference: inter-observer agreement.** Scoring the second observer (`_2ndHO`) against the first gives F1 0.779 and
clDice 0.837 at scale 0.55 ([`second_observer.json`](second_observer.json)).

## Results (all 28 images, threshold 0.5; mean ± seed SD)

| Group | n seeds | F1 @0.55 | AUC @0.55 | F1 @0.75 | clDice @0.75 | F1 @1.0 | AUC @1.0 | clDice @1.0 |
|---|---|---|---|---|---|---|---|---|
| unet | 3 | **0.7688** ± .0019 | **0.9813** ± .0003 | 0.7437 ± .0046 | 0.7726 ± .0079 | 0.5064 ± .0363 | 0.9271 ± .0086 | 0.4778 ± .0492 |
| no-router | 2 | 0.7644 ± .0041 | 0.9800 ± .0005 | 0.7506 ± .0017 | 0.7869 ± .0039 | **0.6097** ± .0010 | 0.9422 ± .0025 | **0.6053** ± .0138 |
| catnet-s | 1 | 0.7678 | 0.9802 | 0.7484 | 0.7851 | 0.5899 | 0.9414 | 0.5926 |
| ac-full | 2 | 0.7661 ± .0034 | 0.9802 ± .0000 | 0.7466 ± .0061 | 0.7850 ± .0032 | 0.5775 ± .0294 | 0.9269 ± .0089 | 0.5774 ± .0147 |
| q-norouter | 1 | 0.7683 | 0.9801 | **0.7550** | 0.7900 | 0.5830 | **0.9424** | 0.5724 |
| q-full | 1 | 0.7655 | 0.9807 | 0.7516 | **0.7905** | 0.5823 | 0.9353 | 0.5848 |

The full paired-test tables are in [`seed_summary.md`](seed_summary.md), with machine-readable copies in
[`seed_summary.json`](seed_summary.json). Per-image metrics, routing reports and oracle thresholds are in
`scale*/<run>/metrics.json`.

### Key paired comparisons (seed-averaged, 28 images)

| Comparison | Scale | ΔF1 | ΔAUC | ΔclDice | Image wins (F1) | Holds for every seed pair? |
|---|---|---|---|---|---|---|
| no-router − unet | 0.55 | −0.0044 | −0.0013 | −0.0020 | 4/28 | AUC only |
| no-router − unet | 0.75 | +0.0069 | +0.0003 | +0.0142 | 20/28 | F1, clDice |
| no-router − unet | 1.0 | **+0.103** | **+0.015** | **+0.128** | 28/28 | all three |
| ac-full − unet | 1.0 | +0.071 | −0.000 | +0.100 | 26/28 | F1, clDice |
| ac-full − no-router | 0.55 | +0.0017 | +0.0002 | +0.0029 | 24/28 | clDice only |
| ac-full − no-router | 1.0 | **−0.032** | **−0.015** | **−0.028** | 0/28 | all three |
| q-full − q-norouter | 1.0 | −0.0006 | −0.0071 | +0.0124 | 13/28 | single seed |

## Interpretation

**1. At matched scale, CHASE is not a robustness test for these models.** With vessel width matched, every model
loses about 0.05 F1 relative to DRIVE. Inter-observer F1 on CHASE is 0.779, so most of that drop is label and annotation
style rather than model failure. The U-Net's small but seed-stable AUC lead mirrors its lead on DRIVE itself.

**2. The CAT-Net advantage is scale robustness, and it is partly calibration.** At native scale every model's
probabilities collapse: the best threshold drops from 0.5 to ≤ 0.10, and for most runs it hits the grid floor of 0.05.
At the oracle threshold:

| | F1 @0.5 | Oracle F1 (t ≤ 0.10) |
|---|---|---|
| U-Net seeds | 0.467–0.539 | 0.606–0.647 |
| no-router seeds | 0.609–0.610 | 0.662–0.680 |

The gap shrinks from about 0.10 to about 0.03 but does not close, and the AUC gap (+0.015, 28/28) confirms a real
discrimination difference. The likely mechanism: global and strip attention at 1/8–1/16 resolution see whole vessel
segments, so a vessel twice as wide still looks like a vessel. The U-Net's purely convolutional receptive field sees
an unfamiliarly wide bright-dark profile. (The oracle is a lower bound for most runs because it is clipped at 0.05.)

**3. The router does not adapt. It saturates.** On DRIVE, and on CHASE at 0.55, `ac-full` routes per pixel with
vessel-driven weights (see [`../drive_ac`](../drive_ac/README.md)). At native scale the mean branch usage
(local / global / strip) becomes:

| Run | Block | Scale 0.55 | Scale 1.0 |
|---|---|---|---|
| `ac-full` | 1/8 | 0.38 / 0.15 / 0.46 | 0.28 / 0.05 / **0.67** |
| `ac-full` | 1/16 | 0.34 / 0.63 / 0.03 | 0.10 / **0.87** / 0.04 |
| `ac-full-seed7` | 1/8 | 0.23 / 0.43 / 0.34 | 0.11 / **0.65** / 0.23 |
| `ac-full-seed7` | 1/16 | 0.48 / 0.17 / 0.35 | **0.69** / 0.05 / 0.26 |
| `q-full` | 1/8 | 0.49 / 0.33 / 0.18 | **0.77** / 0.17 / 0.06 |

Each seed collapses onto a *different* branch, so this is not a principled response to the shift. It is
out-of-distribution router inputs producing near-one-hot softmax outputs. A fixed mix cannot fail this way, which
explains why `no-router` is the most robust variant. The 1/4 block (`q-full`) helps clDice at native scale (+0.012) but
hurts AUC (−0.007). That is the same operating-point shift seen on DRIVE, not better discrimination.

## Caveats

* **Seeds:** 2–3 per group for the main comparisons and 1 for `catnet-s`, `q-full` and `q-norouter`. Under a shift,
  between-seed SD reaches 0.03–0.05 F1, so the single-seed rows at 1.0 should not be ranked against each other.
* **Short CPU training:** all models were trained for 24 short CPU epochs. With full training the U-Net might
  generalise across scale better or worse; scale augmentation during training is the obvious control, and was not run.
* **No scale augmentation:** the robustness gap is a property of models trained *without* scale augmentation.
  Practically, rescaling the input to match vessel width (scale 0.55) beats any architecture choice here by a wide
  margin.
* **Data provenance:** CHASE_DB1 was obtained from a GitHub mirror (`playyalp1/CHASE_DB1`) because the Kingston
  host was not reachable from the build environment. FOV masks are estimated.
* **Test split:** metrics use all 28 images. The 8-image test split alone (`test_split_mean` in each JSON) gives lower
  absolute numbers but the same ordering.

## Reproduce

```bash
python scripts/cross_dataset.py --ckpt runs/drive_cpu/unet/best.pt --name unet \
    --dataset chase --root data/CHASE_DB1 --scales 0.55 0.75 1.0 --tta --out results/chase_xd
# ...repeat for each checkpoint, then:
python scripts/chase_summary.py --root results/chase_xd
```
