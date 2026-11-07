# Cross-dataset: train on DRIVE, test on HRF (zero-shot)

**Question:** on CHASE_DB1 at native resolution, with vessels 1.9× wider than DRIVE's, every CAT-Net beat every
U-Net by about 0.1 F1 ([`../chase_xd`](../chase_xd/README.md)). Does a comparable width shift on a third dataset give
the same result? HRF (High-Resolution Fundus) has 45 images at 3504×2336 px: 15 healthy, 15 glaucoma and 15 diabetic
retinopathy. Its vessels are 9.3 px wide, 2.8× DRIVE's 3.3 px. So HRF can be tested both at matched width (scale 0.35)
and at a 2× shift (scale 0.7) close to CHASE's native case.

**Bottom line**

* **At matched width (scale 0.35) all 16 models tie.** Group means are F1 0.756–0.761 and AUC 0.982–0.984. As on CHASE, the U-Net
  has a small, seed-consistent AUC lead (+0.0017 over the fixed mix, 43/45 images).
* **At a 2× shift (scale 0.7) the CHASE result does *not* repeat.**
  * The fixed-mix CAT-Net and the U-Net tie on F1 (0.707 vs 0.706, p = 0.56).
  * The U-Net has the *better* AUC (+0.006 on 38/45 images, consistent across seeds).
  * The CAT-Net keeps only a small clDice edge (+0.004, consistent across seeds).
  * On CHASE at 1.9× the same comparison gave F1 +0.103 and AUC +0.015 for the CAT-Net.
* **So the CHASE robustness gap is dataset-specific, not a law of vessel width.** On HRF the U-Net degrades
  gracefully: −0.05 F1 from 0.35 to 0.7, against −0.26 on CHASE from 0.55 to 1.0. The CAT-Net's extra context
  therefore has nothing to rescue.
* **At an intermediate 1.4× shift (scale 0.5) everything ties** (F1 0.773–0.779).
* **The adaptive router hurts again under the shift.**
  * `ac-full` is the worst multi-seed model at 0.7: F1 −0.012 vs the fixed mix, and AUC −0.020 vs the U-Net on
    0/45 images. The only model below it is the single-seed adaptive `q-full`.
  * Its routing saturates exactly as on CHASE: 0.76 global or 0.69 local at 1/16, depending on the seed.
* **Scale augmentation fixes the shift for every architecture, and the architectures then tie.**
  * Trained with 0.5–2× zoom ([`../scaleaug`](../scaleaug/README.md)), each model gains about +0.06 F1 at 0.7 on
    45/45 images, reaching F1 0.767–0.768 for all three.
  * The augmented U-Net has the best AUC (+0.003 over the augmented fixed mix, consistent across seeds). F1 and clDice
    differences between the augmented models are not significant.

## Setup

* **Checkpoints:** the same DRIVE-trained checkpoints as [`../chase_xd`](../chase_xd/README.md) and
  [`../stare_xd`](../stare_xd/README.md), plus the scale-augmented models (`-saug`).
* **Data:**
  * all 45 HRF images, with the manual labels (`manual1`) and the supplied FOV masks;
  * obtained from the GitHub mirror `lakshayagarwal23/diabetic-retinopathy-tda`, because the FAU host was not
    reachable from the build environment;
  * LFS files were fetched directly and checked against their sha256 pointers.
* **Scale:**
  * mean vessel width (area / skeleton length) is 9.30 px on HRF and 3.30 px on DRIVE, so matched width is
    **0.35** (DRIVE-like FOV size would be 0.18, but vessel width is what the models are sensitive to);
  * **0.5** and **0.7** give a 1.4× and 2.0× width shift;
  * native scale (2.8×) was not run: with 8 MP images and no GPU, it was too slow and too large for memory.
* **Inference:**
  * sliding window, threshold 0.5, metrics inside the FOV;
  * **no flip TTA**, unlike CHASE and STARE, because of the image size. This is identical for every model;
  * the statistics (seed-averaged paired Wilcoxon, sign-holds-for-every-seed-pair check) are the same as for CHASE.
* **Compute:**
  * the first runs used 1 thread and the rest used 2;
  * the evaluation code was made leaner in memory part-way through, with no change to the numbers: the new code
    reproduces earlier outputs exactly.

## Results (45 images, threshold 0.5, no TTA; mean ± seed SD)

| Group | Seeds | F1 @0.35 | AUC @0.35 | clDice @0.35 | F1 @0.7 | AUC @0.7 | clDice @0.7 |
|---|---|---|---|---|---|---|---|
| unet | 3 | 0.7578 ± .0015 | 0.9834 ± .0002 | 0.8364 ± .0023 | 0.7057 ± .0129 | 0.9534 ± .0046 | 0.7381 ± .0044 |
| no-router | 2 | 0.7570 ± .0066 | 0.9817 ± .0006 | 0.8354 ± .0001 | 0.7068 ± .0068 | 0.9470 ± .0010 | 0.7417 ± .0010 |
| ac-full | 2 | 0.7590 ± .0020 | 0.9823 ± .0000 | 0.8370 ± .0010 | 0.6946 ± .0222 | 0.9336 ± .0117 | 0.7312 ± .0108 |
| catnet-s | 1 | 0.7594 | 0.9823 | **0.8372** | 0.7100 | 0.9495 | 0.7486 |
| q-norouter | 1 | **0.7606** | 0.9824 | 0.8354 | 0.6959 | 0.9509 | 0.7358 |
| q-full | 1 | 0.7571 | 0.9831 | 0.8351 | 0.6874 | 0.9462 | 0.7321 |
| unet-saug | 2 | 0.7585 ± .0011 | **0.9840** ± .0001 | 0.8350 ± .0024 | **0.7678** ± .0035 | **0.9802** ± .0013 | 0.8122 ± .0061 |
| no-router-saug | 2 | 0.7563 ± .0038 | 0.9825 ± .0002 | 0.8322 ± .0002 | 0.7673 ± .0000 | 0.9774 ± .0014 | **0.8137** ± .0033 |
| ac-full-saug | 2 | 0.7562 ± .0029 | 0.9832 ± .0002 | 0.8335 ± .0003 | 0.7675 ± .0016 | 0.9788 ± .0008 | 0.8127 ± .0008 |

**Scale 0.5 (1.4× width shift):**

| Group | Seeds | F1 | AUC | clDice |
|---|---|---|---|---|
| unet | 3 | 0.7755 ± .0018 | 0.9767 ± .0008 | 0.8233 ± .0029 |
| no-router | 2 | 0.7749 ± .0051 | 0.9751 ± .0005 | 0.8240 ± .0000 |
| ac-full | 2 | 0.7728 ± .0028 | 0.9738 ± .0021 | 0.8227 ± .0000 |
| catnet-s | 1 | 0.7766 | 0.9769 | 0.8273 |
| q-norouter | 1 | **0.7793** | 0.9777 | 0.8263 |
| q-full | 1 | 0.7732 | 0.9771 | 0.8260 |
| unet-saug | 2 | 0.7737 ± .0018 | **0.9836** ± .0006 | **0.8379** ± .0039 |
| no-router-saug | 2 | 0.7752 ± .0007 | 0.9818 ± .0005 | 0.8368 ± .0016 |
| ac-full-saug | 2 | 0.7746 ± .0030 | 0.9827 ± .0002 | 0.8374 ± .0006 |

* At 0.5 every group lies within F1 0.773–0.779. No F1 difference against the U-Net is consistent across seeds,
  except the single-seed `q-norouter` (+0.004) and `q-full` (−0.002).
* Scale augmentation does not change F1 at 0.5. It does improve AUC (+0.007 to +0.009) and clDice (+0.013 to
  +0.015) for every model, on 42–45 of 45 images.
* The single-seed variants at 0.7 do not change the picture. `catnet-s` is +0.004 F1 over the U-Net, which is not
  consistent across seeds, and it has the worse AUC. `q-norouter` (−0.010) and `q-full` (−0.018) are below the
  U-Net, and `q-full`, the adaptive 1/4-block variant, is the worst model at 0.7.

All paired tests are in [`seed_summary.md`](seed_summary.md) and [`seed_summary.json`](seed_summary.json).

### Key paired comparisons (seed-averaged, 45 images; image wins on F1)

| Comparison | Scale | ΔF1 | ΔAUC | ΔclDice | Wins | Holds for every seed pair? |
|---|---|---|---|---|---|---|
| no-router − unet | 0.35 | −0.0009 | −0.0017 | −0.0010 | 15/45 | AUC |
| no-router − unet | 0.7 | +0.0011 | **−0.0064** | +0.0036 | 24/45 | AUC, clDice |
| ac-full − unet | 0.7 | −0.0111 | **−0.0198** | −0.0069 | 7/45 | AUC |
| ac-full − no-router | 0.7 | **−0.0122** | −0.0134 | −0.0106 | 1/45 | AUC, clDice |
| unet-saug − unet | 0.7 | **+0.0621** | +0.0268 | +0.0741 | 45/45 | all three |
| no-router-saug − no-router | 0.7 | +0.0605 | +0.0304 | +0.0720 | 45/45 | all three |
| no-router-saug − unet-saug | 0.7 | −0.0005 | −0.0028 | +0.0015 | 27/45 | AUC |
| ac-full-saug − no-router-saug | 0.7 | +0.0002 | +0.0014 | −0.0011 | 20/45 | none |

## Interpretation

**1. The CAT-Net advantage tracks how badly the U-Net fails, not the width shift itself.** Fixed-mix CAT-Net minus
U-Net, all three datasets (seed-averaged):

| Dataset @ scale | Width vs DRIVE | U-Net F1 | ΔF1 | ΔAUC | ΔclDice |
|---|---|---|---|---|---|
| HRF @0.35 | 0.99× | 0.758 | −0.001 | −0.002 | −0.001 |
| HRF @0.5 | 1.41× | 0.776 | −0.001 | −0.002 | +0.001 |
| CHASE @0.55 | 1.05× | 0.769 | −0.004 | −0.001 | −0.002 |
| STARE @0.8 | 1.04× | 0.785 | −0.004 | −0.000 | +0.002 |
| STARE @1.0 | 1.30× | 0.802 | −0.002 | −0.000 | +0.002 |
| CHASE @0.75 | 1.43× | 0.744 | +0.007 | +0.000 | +0.014 |
| HRF @0.7 | 1.97× | 0.706 | +0.001 | **−0.006** | +0.004 |
| CHASE @1.0 | 1.91× | **0.506** | **+0.103** | **+0.015** | **+0.128** |

* HRF at 0.7 and CHASE at 1.0 have the same width shift.
* HRF differs from CHASE in other ways: much higher resolution, a different camera, and adult eyes rather than
  children's. This analysis does not tell us which of these makes the CHASE images harder for the U-Net.
* The U-Net loses 0.05 F1 on HRF and 0.26 on CHASE. A large CAT-Net advantage appeared only in the case where the
  U-Net collapsed, so width alone does not predict it.

**2. Under the shift, all unaugmented models lose sensitivity and calibration, by a similar amount.**

* At 0.7, sensitivity falls to 0.63–0.69, against 0.91–0.92 at 0.35.
* The oracle threshold drops to 0.12–0.24 for every unaugmented run, and oracle F1 rises only to 0.69–0.72.
* So wide vessels are partly missed, and partly still detected but with low confidence. The same happens on CHASE,
  but there the oracle threshold reached the grid floor of 0.05.
* With scale augmentation the oracle threshold at 0.7 is 0.80–0.86, the same over-segmentation as at matched scale.

**3. Matched-scale HRF is a label-style shift, as on STARE.** At 0.35 every model over-segments relative to the HRF
labels: sensitivity 0.91–0.92, precision 0.64–0.66, and an oracle threshold of 0.90–0.94 (oracle F1 0.790–0.802).
This affects all architectures equally.

**4. The router saturates again.** Mean branch usage (local / global / strip):

| Run | Block | Scale 0.35 | Scale 0.7 |
|---|---|---|---|
| `ac-full` | 1/8 | 0.29 / 0.25 / 0.45 | 0.15 / 0.14 / **0.71** |
| `ac-full` | 1/16 | 0.43 / 0.52 / 0.05 | 0.15 / **0.76** / 0.09 |
| `ac-full-seed7` | 1/8 | 0.26 / 0.44 / 0.30 | 0.13 / **0.69** / 0.18 |
| `ac-full-seed7` | 1/16 | 0.47 / 0.19 / 0.34 | **0.69** / 0.07 / 0.24 |
| `ac-full-saug` | 1/16 | 0.28 / 0.54 / 0.19 | 0.24 / **0.66** / 0.09 |
| `ac-full-saug-seed7` | 1/16 | 0.45 / 0.27 / 0.27 | **0.63** / 0.12 / 0.25 |

* The pattern is the same as on CHASE: each seed saturates onto its own preferred branch, global for seed 42 and
  local for seed 7, and on the same branch on both datasets.
* Without augmentation this costs accuracy. Both adaptive models are the weakest at 0.7: `ac-full` among the
  multi-seed groups, and `q-full` overall.
* With augmentation the lean is milder and harmless (`ac-full-saug` ties the fixed mix).

## Caveats

* **No TTA** on HRF, unlike CHASE and STARE. Absolute numbers are therefore not directly comparable across datasets;
  comparisons between models within HRF are.
* **Native scale (2.8× width) was not run.** At 0.7 the shift (2.0×) already matches CHASE native.
* **Seeds:** 2–3 per main group. At 0.7 the unaugmented seed SD is 0.007–0.022 F1, comparable to the F1 differences
  between architectures.
* **Short CPU training**, as everywhere in this repository.

## Reproduce

```bash
python scripts/cross_dataset.py --ckpt runs/drive_cpu/unet/best.pt --name unet \
    --dataset hrf --root data/HRF --scales 0.35 0.5 0.7 --out results/hrf_xd
# ...repeat for each checkpoint (no --tta), then:
python scripts/chase_summary.py --root results/hrf_xd
```
