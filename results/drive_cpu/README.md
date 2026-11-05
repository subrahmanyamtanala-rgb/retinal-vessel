# DRIVE: first real-data experiment (reduced CPU budget)

**Bottom line:** at this budget, CAT-Net-S does **not** beat a size-matched U-Net. It is slightly but consistently
worse. The likely cause is that the context router **collapses** to one fixed branch per stage instead of adapting
per pixel. The results are reported as measured.

## Setup

**Data:** DRIVE, official split.

* 20 training images, of which 2 are held out for validation (seed 42). The checkpoint with the best validation F1 is
  kept.
* 20 test images.
* Metrics are computed inside the FOV mask, at threshold 0.5, against the 1st manual annotator.

**Hardware:** 4-core CPU, no GPU. Each model trained on one thread, all three in parallel.

**Schedule:** identical for all models.

* 24 epochs × 1000 random 128 px patches = **24 k patches**. That is roughly **3 % of the full schedule** in
  [`configs/drive.json`](../../configs/drive.json) (200 × 4000 = 800 k).
* Batch size 8, AdamW, learning rate 1e-3, cosine schedule, 2 warm-up epochs.
* Loss: BCE + Dice + 0.5·clDice.

**Inference:** 256 px sliding window, 50 % overlap, flip test-time augmentation (×4).

| Run | Model | Params |
|---|---|---|
| `unet` | residual U-Net, widths 20–288 | 2.92 M |
| `catnet-s-no-router` | CAT-Net-S with fixed equal branch weights | 2.91 M |
| `catnet-s` | CAT-Net-S (full) | 2.94 M |

## Results (20 DRIVE test images, mean over images, flip TTA)

| Model | Acc | Se | Sp | F1 | IoU | MCC | AUC | AP | clDice |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | **0.9538** | **0.8464** | 0.9699 | **0.8227** | **0.6991** | **0.7979** | **0.9801** | **0.9128** | **0.8458** |
| CAT-Net-S, no router | **0.9538** | 0.8395 | **0.9709** | 0.8215 | 0.6973 | 0.7965 | 0.9790 | 0.9104 | 0.8409 |
| CAT-Net-S | 0.9535 | 0.8413 | 0.9702 | 0.8207 | 0.6962 | 0.7956 | 0.9790 | 0.9101 | 0.8418 |

Without TTA, the F1 scores are: U-Net 0.8208, no-router 0.8203, CAT-Net-S 0.8187.

### Paired comparison per image (Wilcoxon signed-rank, n = 20)

| Comparison | Metric | Mean difference | Images won | p |
|---|---|---|---|---|
| CAT-Net-S vs U-Net | F1 | −0.0020 | 2 / 20 | < 0.001 |
| CAT-Net-S vs U-Net | AUC | −0.0012 | 0 / 20 | < 0.001 |
| CAT-Net-S vs U-Net | clDice | −0.0040 | 3 / 20 | < 0.001 |
| CAT-Net-S vs no-router | F1 | −0.0008 | 7 / 20 | 0.044 |
| CAT-Net-S vs no-router | AUC | −0.0000 | 11 / 20 | 0.78 |
| CAT-Net-S vs no-router | clDice | +0.0009 | 16 / 20 | 0.009 |

The differences are small in absolute terms (≤ 0.4 points) but consistent across images. These are single-seed runs,
so the p-values cover image-to-image variation only, not training-seed variation.

## Diagnosis: the router collapses

Mean branch weights over the 20 test images:

| Block | Local window | Global grid | Directional strip |
|---|---|---|---|
| `enc4.0` (1/8) | **0.994** | 0.003 | 0.003 |
| `enc5.0` (1/16) | 0.013 | **0.987** | 0.001 |

The router learns a near-hard, **image-independent** choice: local attention at 1/8 and global attention at 1/16.
The routing map in `figures/` is uniformly red across the FOV, with no visible per-pixel structure. The strip branch
is essentially switched off.

The model is therefore behaving as a plain hybrid with one mixer per stage, and the per-pixel adaptivity that
motivates the design never shows up. Nothing in the loss rewards a mixture, and with a softmax gate the easiest way
down the loss is to commit to the branch that trains fastest. This is the same failure as expert collapse in
mixture-of-experts models.

## What to change next

1. **Keep the router from collapsing:**
   * an entropy or load-balancing penalty on the routing weights;
   * a router temperature that starts high and is annealed;
   * branch dropout, which randomly zeroes one branch during training.
2. **Make routing conditional on content**, by feeding it a vesselness or contrast cue rather than only the features.
3. **Use the full schedule on a GPU.** Transformer models typically need a longer schedule than CNNs to catch up, so
   the 3 % budget here may favour the U-Net. The comparison should be repeated at the full schedule, with ≥ 3 seeds,
   before drawing conclusions about the architecture.
4. **Add cross-dataset tests** (DRIVE → CHASE_DB1 / STARE). Adaptivity to image-level context should matter most
   under domain shift, which this in-domain test does not probe.

## Files

* `<run>/config.json`, `history.json`, `train.log`: training configuration and logs.
* `<run>/metrics.json`: per-image and mean test metrics with TTA.
* `<run>/test_metrics_no_tta.json`: the same metrics without TTA.
* `figures/0{1,2}_routing.png`: input | CAT-Net-S prediction | routing map at 1/8, coloured R = local, G = global,
  B = strip.

To reproduce, follow the same steps as in the repository README, with the overrides
`--epochs 24 --samples-per-epoch 1000 --batch-size 8 --warmup-epochs 2 --val-every 4 --threads 1`.
