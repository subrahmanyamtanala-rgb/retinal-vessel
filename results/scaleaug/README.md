# Control: scale augmentation during training

**Question:** on CHASE_DB1 at native resolution, every CAT-Net beat every U-Net by about 0.1 F1
([`../chase_xd`](../chase_xd/README.md)). Is that an architectural advantage, or just what happens to a model that
never saw wide vessels in training? The control: retrain the three main models with **random zoom**, so that
training patches contain vessels from about half to twice their DRIVE width. Then repeat every cross-dataset test.

**Bottom line**

* **Scale augmentation removes the U-Net's scale failure completely.** On CHASE at native scale (vessels 1.9× wider
  than DRIVE's), the augmented U-Net scores F1 0.758, against 0.506 without augmentation: +0.251, on 28/28 images,
  for every seed pair. The augmented models lose only 0.011–0.014 F1 between matched and native scale; without
  augmentation the U-Net lost 0.26.
* **With augmentation, the CAT-Net advantage disappears and reverses.** The augmented U-Net beats both augmented
  CAT-Nets on CHASE at every scale (F1 +0.004 to +0.007, 22–27 of 28 images) and on STARE at 0.8 (F1 +0.004). Every
  one of these differences keeps its sign for all seed pairs. On STARE at native scale the differences are small and
  not consistent across seeds.
* **So the CAT-Net "robustness" was a substitute for data augmentation, not an extra.** Global and strip attention made
  the unaugmented CAT-Nets degrade more gracefully. Once the training data cover the scale range, that edge is gone, and
  the U-Net's small in-distribution lead returns.
* **Augmentation fixes the calibration collapse.** Without it, the best threshold at CHASE native scale fell to
  0.05–0.10. With it, the best threshold stays at 0.48–0.63, and F1 at 0.5 is within 0.002 of the oracle F1.
* **The router still leans heavily, but no longer hurts.** At CHASE native scale, `ac-full-saug` still puts 0.64–0.73
  of its 1/16 weight on a single branch (a different branch per seed). However, it is no longer worse than the fixed
  mix in a seed-consistent way (F1 −0.002).
* **Cost on DRIVE:** F1 −0.001 to −0.002 and clDice −0.011 to −0.017. A likely cause is that zoomed-out patches
  shrink capillaries below one pixel, which costs skeleton accuracy on DRIVE's thin vessels.
* **Augmentation also helps on STARE.** F1 improves for every model, by +0.016 to +0.018 at scale 0.8, which is
  seed-consistent. On CHASE at matched scale only the U-Net gains (F1 +0.003). The CAT-Nets are unchanged on F1 and
  lose 0.005–0.009 clDice.

## Setup

* **Models:** `unet`, `catnet-s-no-router` and `ac-full` (`catnet-s-ac` with the anti-collapse recipe), with the same
  24-epoch CPU schedule as every other run. Seeds 42 and 7 for each.
* **Augmentation:**
  * each training patch is zoomed by a factor `z`, log-uniform in [0.5, 2]
    (`--scale-aug-min 0.5 --scale-aug-max 2.0`);
  * a crop of side `round(128 / z)` is resized to 128 px: bilinear for the image, bilinear > 0.5 for the label,
    nearest for the FOV;
  * mean vessel width in the zoomed patches is 2.2, 3.7 and 7.3 px at z = 0.5, 1 and 2;
  * nothing else changes: same flips, rotations, intensity jitter and vessel-biased crop sampling.
* **Evaluation:** identical to the unaugmented runs ([`../chase_xd`](../chase_xd/README.md),
  [`../stare_xd`](../stare_xd/README.md)): all images, flip TTA, threshold 0.5, seed-averaged paired Wilcoxon tests,
  and a check that each sign holds for every seed pair. Group names carry the suffix `-saug`. The tests are in
  `../{chase,stare,hrf}_xd/seed_summary.md`.

## DRIVE (20 test images, flip TTA)

| Model | Without augmentation (seeds) | F1 | AUC | clDice | With 0.5–2× zoom (2 seeds) | F1 | AUC | clDice |
|---|---|---|---|---|---|---|---|---|
| U-Net | 3 | **0.8228** | **0.9800** | **0.8455** | | **0.8210** | **0.9787** | 0.8288 |
| CAT-Net-S, fixed mix | 2 | 0.8204 | 0.9789 | 0.8419 | | 0.8193 | 0.9780 | 0.8304 |
| CAT-Net-S `ac-full` | 2 | 0.8208 | 0.9791 | 0.8424 | | 0.8198 | 0.9782 | **0.8315** |

Per-seed values: [`*/metrics.json`](.) (TTA) and `*/test_metrics_no_tta.json`. Without augmentation the U-Net leads
on clDice; with augmentation it falls slightly behind the CAT-Nets, since it loses the most clDice.

## Cross-dataset (zero-shot), F1 / AUC / clDice (mean over seeds)

| Dataset @ scale | Width vs DRIVE | U-Net | U-Net + aug | Fixed mix | Fixed mix + aug | `ac-full` | `ac-full` + aug |
|---|---|---|---|---|---|---|---|
| CHASE @0.55 | 1.05× | 0.769 / .9813 / .842 | **0.772** / **.9818** / .842 | 0.764 / .9800 / .840 | 0.765 / .9801 / .836 | 0.766 / .9802 / **.843** | 0.766 / .9804 / .835 |
| CHASE @0.75 | 1.43× | 0.744 / .9711 / .773 | **0.776** / **.9797** / **.830** | 0.751 / .9714 / .787 | 0.770 / .9780 / .821 | 0.747 / .9700 / .785 | 0.769 / .9779 / .817 |
| CHASE @1.0 | 1.91× | 0.506 / .9271 / .478 | **0.758** / **.9740** / **.785** | 0.610 / .9422 / .605 | 0.754 / .9724 / .776 | 0.578 / .9269 / .577 | 0.752 / .9720 / .773 |
| STARE @0.8 | 1.04× | 0.785 / .9882 / .868 | **0.803** / **.9890** / **.874** | 0.782 / .9880 / .870 | 0.799 / .9885 / .871 | 0.784 / .9878 / .871 | 0.800 / .9885 / .872 |
| STARE @1.0 | 1.30× | 0.802 / .9882 / .871 | **0.807** / **.9899** / **.874** | 0.800 / .9879 / .872 | 0.806 / .9893 / .873 | 0.802 / .9874 / **.874** | 0.805 / .9894 / .871 |

HRF: see [below](#hrf).

### Key paired comparisons (seed-averaged; image wins on F1)

| Comparison | Dataset @ scale | ΔF1 | ΔAUC | ΔclDice | Wins | Holds for every seed pair? |
|---|---|---|---|---|---|---|
| U-Net + aug − U-Net | CHASE @1.0 | **+0.251** | +0.047 | +0.307 | 28/28 | all three |
| fixed mix + aug − fixed mix | CHASE @1.0 | +0.144 | +0.030 | +0.170 | 28/28 | all three |
| `ac-full` + aug − `ac-full` | CHASE @1.0 | +0.175 | +0.045 | +0.196 | 28/28 | all three |
| fixed mix + aug − U-Net + aug | CHASE @0.55 | −0.0068 | −0.0017 | −0.0065 | 1/28 | all three |
| fixed mix + aug − U-Net + aug | CHASE @0.75 | −0.0058 | −0.0017 | −0.0093 | 2/28 | all three |
| fixed mix + aug − U-Net + aug | CHASE @1.0 | **−0.0036** | −0.0016 | −0.0096 | 6/28 | all three |
| `ac-full` + aug − U-Net + aug | CHASE @1.0 | −0.0052 | −0.0019 | −0.0120 | 5/28 | all three |
| fixed mix + aug − U-Net + aug | STARE @0.8 | −0.0039 | −0.0005 | −0.0029 | 4/20 | F1, AUC |
| fixed mix + aug − U-Net + aug | STARE @1.0 | −0.0009 | −0.0006 | −0.0013 | 6/20 | AUC |
| `ac-full` + aug − fixed mix + aug | CHASE @1.0 | −0.0016 | −0.0004 | −0.0024 | 12/28 | none |

## Interpretation

**1. The unaugmented U-Net failure was distribution shift, not missing context.** Without augmentation, the U-Net's
probabilities collapsed at CHASE native scale: oracle threshold 0.05, the grid floor, for all 3 seeds. With
augmentation, the oracle threshold is 0.54–0.55 and oracle F1 equals F1 at 0.5 to within 0.001:

| CHASE @1.0 | F1 @0.5 | Oracle threshold | Oracle F1 |
|---|---|---|---|
| U-Net (3 seeds) | 0.467–0.539 | 0.05 | 0.601–0.643 |
| U-Net + aug (2 seeds) | 0.757–0.759 | 0.54–0.55 | 0.757–0.759 |
| fixed mix (2 seeds) | 0.609–0.610 | 0.06–0.10 | 0.659–0.677 |
| fixed mix + aug (2 seeds) | 0.753–0.755 | 0.48–0.60 | 0.753–0.757 |

Even the oracle cannot rescue the unaugmented models, so augmentation also restores discrimination (AUC 0.927 →
0.974 for the U-Net), not just calibration.

**2. Under matched training data, the convolutional baseline is at least as good.** This mirrors DRIVE itself, where
the U-Net leads at this budget. Global attention gives the CAT-Nets a wider effective receptive field, which helped
when test vessels looked unlike anything in training. Augmentation makes them look familiar, and then the extra
context buys nothing.

**3. The router's lean is a seed trait, made more extreme by shift.** Mean branch usage (local / global / strip):

| Run | Block | CHASE @0.55 | CHASE @1.0 | STARE @1.0 |
|---|---|---|---|---|
| `ac-full` | 1/16 | 0.34 / 0.63 / 0.03 | 0.10 / **0.87** / 0.04 | 0.48 / 0.48 / 0.04 |
| `ac-full` + aug | 1/16 | 0.16 / 0.63 / 0.21 | 0.13 / **0.73** / 0.14 | 0.35 / 0.48 / 0.16 |
| `ac-full-seed7` | 1/16 | 0.48 / 0.17 / 0.35 | **0.69** / 0.05 / 0.26 | 0.45 / 0.23 / 0.32 |
| `ac-full-seed7` + aug | 1/16 | 0.47 / 0.28 / 0.25 | **0.64** / 0.12 / 0.24 | 0.42 / 0.32 / 0.25 |

* Augmentation softens the saturation (0.87 → 0.73 and 0.69 → 0.64) but does not remove it.
* Each seed leans towards the *same* branch with and without augmentation, which is set by the initialisation.
* The lean no longer costs accuracy (`ac-full` + aug vs fixed mix + aug at CHASE @1.0: F1 −0.002, not
  seed-consistent). Routing still gives no measurable benefit.

## HRF

45 images, no TTA (see [`../hrf_xd`](../hrf_xd/README.md)). F1 / AUC / clDice, mean over seeds:

| Scale | Width vs DRIVE | U-Net | U-Net + aug | Fixed mix | Fixed mix + aug | `ac-full` | `ac-full` + aug |
|---|---|---|---|---|---|---|---|
| 0.35 | 0.99× | 0.758 / .9834 / **.836** | **0.759** / **.9840** / .835 | 0.757 / .9817 / .835 | 0.756 / .9825 / .832 | **0.759** / .9823 / **.837** | 0.756 / .9832 / .834 |
| 0.5 | 1.41× | **0.776** / .9767 / .823 | 0.774 / **.9836** / **.838** | 0.775 / .9751 / .824 | 0.775 / .9818 / .837 | 0.773 / .9738 / .823 | 0.775 / .9827 / .837 |
| 0.7 | 1.97× | 0.706 / .9534 / .738 | **0.768** / **.9802** / .812 | 0.707 / .9470 / .742 | 0.767 / .9774 / **.814** | 0.695 / .9336 / .731 | 0.768 / .9788 / .813 |

* **Augmentation removes the shift penalty for every architecture.** At 0.7 it adds about 0.06 F1 on 45/45 images
  (U-Net +0.062, fixed mix +0.061, `ac-full` +0.073). The augmented models score *higher* at 0.7 than at the matched
  0.35, through better precision at 0.5.
* **The augmented models then tie on F1 and clDice**, with differences of 0.0005 F1 or less and no significant
  difference. The augmented U-Net keeps a seed-consistent AUC lead over the augmented fixed mix (+0.003).
* **At matched scale augmentation does little**, as expected: F1 changes by less than 0.003. At 0.5 (1.4×) F1 is
  also unchanged, but AUC (+0.007 to +0.009) and clDice (+0.013 to +0.015) improve for every model.
* HRF agrees with CHASE: once training covers the scale range, the architectures are equivalent or the U-Net is
  slightly ahead. Unlike CHASE, HRF also shows no large CAT-Net advantage *without* augmentation. The unaugmented
  U-Net degrades only moderately on HRF.

## Caveats

* **Budget:** 24 short CPU epochs. Augmentation makes each patch harder; a longer schedule might recover the DRIVE
  clDice cost, and could change the ranking either way.
* **Seeds:** 2 seeds per augmented model, against 2–3 for the baselines. Most of the key differences hold for every
  seed pairing. However, the augmented-model gaps are small (0.004–0.007 F1), about 2–5× the seed SD.
* **One augmentation range:** only 0.5–2× was tried. The 1.9–2.0× shifts tested (CHASE native, HRF at 0.7) sit
  near the edge of that range. HRF at native scale (2.8×) would fall outside it and was not run.
* **Same evaluation code** as all other cross-dataset runs. The memory-saving changes made for HRF reproduce earlier
  outputs exactly: a full STARE rerun matched, and so did the AUC on random tied data.

## Reproduce

```bash
python -m catnet.train --config runs/scaleaug/unet.json   # = DRIVE config + --scale-aug-min 0.5 --scale-aug-max 2.0
python -m catnet.evaluate --ckpt runs/scaleaug/unet/best.pt --dataset drive --root data/DRIVE --tta --out results/scaleaug/unet
python scripts/cross_dataset.py --ckpt runs/scaleaug/unet/best.pt --name unet-saug \
    --dataset chase --root data/CHASE_DB1 --scales 0.55 0.75 1.0 --tta --out results/chase_xd
python scripts/chase_summary.py --root results/chase_xd
```

The training configs are in `*/config.json`.
