# Cross-dataset: train on DRIVE, test on STARE (zero-shot)

**Question:** does the CHASE_DB1 picture ([`../chase_xd`](../chase_xd/README.md)) hold on a second unseen
dataset? STARE (20 images, 700×605) has more pathology than DRIVE: 10 of its 20 labelled images are abnormal. Its
geometry is closer to DRIVE's: vessels are 1.3× wider, compared with 1.9× on CHASE.

**Bottom line**

* **All models are tied on STARE.**
  * F1 at native scale is 0.797–0.803 and AUC is 0.987–0.988 for every group.
  * No F1 or AUC difference between the U-Net and any CAT-Net is both significant and consistent across seeds,
    except that `q-full` is slightly worse (F1 −0.005).
* **CAT-Nets have a small, consistent connectivity edge.** clDice is +0.002 to +0.005 over the U-Net at both scales.
  For `catnet-s` and `ac-full` the difference is significant and holds for every seed pair.
* **The large CHASE robustness gap does not appear, as predicted by the size of the shift.** STARE at native scale is
  only a 1.3× vessel-width shift. On CHASE a 1.4× shift (scale 0.75) gave a small gap, and only a 1.9× shift
  (native) gave a large one. HRF ([`../hrf_xd`](../hrf_xd/README.md)) tests the 2× case on a third dataset.
* **The router does not saturate here.** Branch usage stays mixed (no branch > 0.6) at both scales and for both
  `ac-full` seeds. On CHASE at native scale the router collapsed onto one branch. Router failure is a large-shift
  phenomenon.
* **The adaptive router is again no better than a fixed branch mix.** `ac-full` − `no-router`: F1 +0.002 (not
  consistent across seeds), AUC −0.0006. `q-full` is worse than `q-norouter` on F1 and clDice at both scales.

## Setup

* **Checkpoints and protocol:** the same 10 DRIVE-trained checkpoints and protocol as
  [`../chase_xd`](../chase_xd/README.md): flip TTA, threshold 0.5, metrics inside the FOV, all 20 images, and
  seed-averaged paired Wilcoxon tests.
* **Reference labels:** Hoover (`labels-ah`). The second observer is Kouznetsova (`labels-vk`).
* **FOV:** estimated from the red channel with threshold 0.18. The default of 0.08 leaked into STARE's noisy dark
  border on im0044 and im0240. CHASE keeps 0.08.
* **Scale:**
  * Mean vessel width is 3.30 px on DRIVE and 4.29 px on STARE.
  * FOV diameter is 538 vs 650 px.
  * So the matched scale is 0.77 (from vessel width) to 0.83 (from FOV diameter). We use **0.8** as matched and
    **1.0** as native.
* **Data provenance:** STARE was obtained from the GitHub mirror `oskarkaluziak/stare-images` (20 images, both label
  sets) because the Clemson host was not reachable from the build environment.

**Reference: inter-observer agreement.** Kouznetsova's labels scored against Hoover's give F1 0.740, Se 0.895,
precision 0.643 and clDice 0.715 at native scale ([`second_observer.json`](second_observer.json)). Kouznetsova marks
far more thin vessels than Hoover, so this is a loose reference, not a ceiling. The models agree with Hoover better
than the second observer does.

## Results (all 20 images, threshold 0.5; mean ± seed SD)

| Group | n seeds | F1 @0.8 | AUC @0.8 | clDice @0.8 | F1 @1.0 | AUC @1.0 | clDice @1.0 |
|---|---|---|---|---|---|---|---|
| unet | 3 | **0.7853** ± .0015 | **0.9882** ± .0002 | 0.8676 ± .0008 | 0.8018 ± .0009 | **0.9882** ± .0002 | 0.8705 ± .0014 |
| no-router | 2 | 0.7815 ± .0051 | 0.9880 ± .0001 | 0.8700 ± .0001 | 0.7998 ± .0047 | 0.9879 ± .0001 | 0.8722 ± .0014 |
| catnet-s | 1 | 0.7852 | 0.9880 | **0.8719** | **0.8033** | **0.9882** | **0.8750** |
| ac-full | 2 | 0.7838 ± .0016 | 0.9878 ± .0000 | 0.8712 ± .0005 | 0.8017 ± .0007 | 0.9874 ± .0001 | 0.8740 ± .0008 |
| q-norouter | 1 | 0.7839 | 0.9881 | 0.8693 | 0.8021 | 0.9878 | 0.8726 |
| q-full | 1 | 0.7792 | 0.9881 | 0.8666 | 0.7969 | 0.9877 | 0.8683 |

The full paired-test tables are in [`seed_summary.md`](seed_summary.md) and [`seed_summary.json`](seed_summary.json).

### Key paired comparisons (seed-averaged, 20 images)

| Comparison | Scale | ΔF1 | ΔAUC | ΔclDice (p) | Holds for every seed pair? |
|---|---|---|---|---|---|
| catnet-s − unet | 0.8 | −0.0001 | −0.0001 | +0.0043 (0.002) | clDice |
| catnet-s − unet | 1.0 | +0.0014 | −0.0000 | +0.0045 (0.003) | F1, clDice |
| ac-full − unet | 0.8 | −0.0015 | −0.0004 | +0.0036 (0.002) | AUC, clDice |
| ac-full − unet | 1.0 | −0.0002 | −0.0008 | +0.0034 (0.03) | AUC, clDice |
| no-router − unet | 1.0 | −0.0020 | −0.0002 | +0.0017 (0.67) | none |
| ac-full − no-router | 1.0 | +0.0019 | −0.0006 | +0.0018 (0.12) | AUC, clDice |
| q-full − q-norouter | 1.0 | −0.0052 | −0.0000 | −0.0043 (0.002) | single seed |

## Interpretation

**1. STARE is mostly a label-style shift, not a scale shift.** Every model over-segments relative to Hoover's
conservative labels:

* sensitivity is about 0.92–0.94 and precision about 0.67–0.72;
* the oracle threshold is about 0.86–0.93 for every run, far above 0.5;
* at the oracle threshold all runs reach F1 0.821–0.828, and the spread between models shrinks further.

The shift is the same for every architecture, so it does not separate them. Native scale scores *higher* at 0.5 than
the "matched" 0.8 (F1 about 0.80 vs 0.78), mainly through higher precision. At the oracle threshold both scales are
equal, so this is an operating-point effect and not a sign that 0.8 is the wrong scale.

**2. The robustness gap scales with the vessel-width shift.** Combining both datasets, CAT-Net minus U-Net
(fixed-mix CAT-Net, seed-averaged):

| Dataset @ scale | Width relative to DRIVE | ΔF1 | ΔclDice |
|---|---|---|---|
| CHASE @0.55 | 1.05× | −0.004 | −0.002 |
| STARE @0.8 | 1.04× | −0.004 | +0.002 |
| STARE @1.0 | 1.30× | −0.002 | +0.002 |
| CHASE @0.75 | 1.43× | +0.007 | +0.014 |
| CHASE @1.0 | 1.91× | +0.103 | +0.128 |

Near the training scale the U-Net is as good or marginally better. The transformer context pays off only as the
apparent vessel size moves well outside the training distribution.

**3. Routing stays healthy under a small shift.** Mean branch usage (local / global / strip):

| Run | Block | Scale 0.8 | Scale 1.0 |
|---|---|---|---|
| `ac-full` | 1/8 | 0.37 / 0.32 / 0.31 | 0.30 / 0.28 / 0.42 |
| `ac-full` | 1/16 | 0.60 / 0.37 / 0.03 | 0.48 / 0.48 / 0.04 |
| `ac-full-seed7` | 1/8 | 0.34 / 0.31 / 0.35 | 0.29 / 0.41 / 0.30 |
| `ac-full-seed7` | 1/16 | 0.37 / 0.31 / 0.32 | 0.45 / 0.23 / 0.32 |

There is no collapse onto one branch, unlike CHASE at native scale (0.87 global or 0.69 local at 1/16). Even so,
routing gives no accuracy benefit over a fixed mix.

## Caveats

* **Small data:** 20 images, half of them pathological. Paired tests have limited power, and the clDice effects
  (+0.003 to +0.005) are small in absolute terms.
* **Seeds:** 1 seed for `catnet-s`, `q-full` and `q-norouter`. Seed SD on STARE is about 0.001–0.005 F1, comparable
  to most of the differences above.
* **Training:** short CPU training (24 epochs) with no scale augmentation, as in all other experiments.
* **FOV masks** are estimated, and the threshold was tuned by eye per dataset. All models share the same masks.

## Reproduce

```bash
python scripts/cross_dataset.py --ckpt runs/drive_cpu/unet/best.pt --name unet \
    --dataset stare --root data/STARE --scales 0.8 1.0 --tta --out results/stare_xd
# ...repeat for each checkpoint, then:
python scripts/chase_summary.py --root results/stare_xd
```
