import json

import numpy as np
import torch

from catnet.data import PatchDataset, list_samples, load_split, preprocess, synthetic_fundus, write_synthetic_dataset
from catnet.inference import sliding_window
from catnet.models import build_model


def test_synthetic_fundus():
    rgb, ves, fov = synthetic_fundus(128, seed=3)
    assert rgb.shape == (128, 128, 3) and rgb.dtype == np.uint8
    assert 0.02 < ves.mean() < 0.3
    assert (ves <= fov).all()


def test_preprocess_range_and_background():
    rgb, _, fov = synthetic_fundus(96, seed=1)
    g = preprocess(rgb, fov)
    assert g.dtype == np.float32 and g.shape == (96, 96)
    assert g.min() >= 0 and g.max() <= 1
    assert (g[fov < 0.5] == 0).all()


def test_folder_dataset_and_patches(tmp_path):
    write_synthetic_dataset(tmp_path, n_train=3, n_test=2, size=96)
    assert len(list_samples("folder", tmp_path, "train")) == 3
    items = load_split("folder", tmp_path, "test")
    assert len(items) == 2 and items[0]["image"].shape == (96, 96)
    ds = PatchDataset(load_split("folder", tmp_path, "train"), patch=48, samples_per_epoch=5)
    x, y, m = ds[0]
    assert x.shape == y.shape == m.shape == (1, 48, 48)
    x2, _, _ = ds[0]
    assert torch.equal(x, x2)  # deterministic given (seed, epoch, index)


def test_scale_augmentation(tmp_path):
    write_synthetic_dataset(tmp_path, n_train=2, n_test=1, size=128)
    items = load_split("folder", tmp_path, "train")
    base = PatchDataset(items, patch=48, samples_per_epoch=40, augment=False)
    ref = PatchDataset(items, patch=48, samples_per_epoch=40, augment=False, scale_range=(1.0, 1.0))
    assert all(torch.equal(base[i][0], ref[i][0]) for i in range(5))  # (1, 1) = unchanged behaviour
    zin = PatchDataset(items, patch=48, samples_per_epoch=40, augment=False, scale_range=(2.0, 2.0))
    zout = PatchDataset(items, patch=48, samples_per_epoch=40, augment=False, scale_range=(0.5, 0.5))
    for ds in (zin, zout):
        x, y, m = ds[0]
        assert x.shape == y.shape == m.shape == (1, 48, 48)
        assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    # zooming in by 2x roughly doubles the vessel fraction of vessel-centred crops vs zooming out by 2x
    frac = lambda ds: float(torch.stack([ds[i][1] for i in range(40)]).mean())
    assert frac(zin) > 1.5 * frac(zout)


def test_drive_layout_discovery(tmp_path):
    from PIL import Image

    for split, idx in (("training", "21"), ("test", "01")):
        for sub in ("images", "1st_manual", "mask"):
            (tmp_path / split / sub).mkdir(parents=True)
        rgb, ves, fov = synthetic_fundus(64, seed=0)
        Image.fromarray(rgb).save(tmp_path / split / "images" / f"{idx}_{split}.tif")
        Image.fromarray((ves * 255).astype(np.uint8)).save(tmp_path / split / "1st_manual" / f"{idx}_manual1.gif")
        Image.fromarray((fov * 255).astype(np.uint8)).save(tmp_path / split / "mask" / f"{idx}_{split}_mask.gif")
    tr = list_samples("drive", tmp_path, "train")
    te = list_samples("drive", tmp_path, "test")
    assert [s.name for s in tr] == ["21"] and [s.name for s in te] == ["01"]
    assert tr[0].fov is not None


def test_sliding_window_shapes():
    m = build_model("catnet-s").eval()
    img = np.random.rand(100, 140).astype(np.float32)
    p = sliding_window(m, img, patch=64, overlap=0.5, tta=True)
    assert p.shape == (100, 140) and np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()


def test_train_smoke(tmp_path):
    from catnet.train import main

    res = main(["--dataset", "synthetic", "--model", "catnet-s", "--epochs", "2", "--samples-per-epoch", "16",
                "--batch-size", "4", "--patch", "64", "--eval-patch", "96", "--workers", "0", "--val-every", "1",
                "--out", str(tmp_path / "run"), "--device", "cpu"])
    assert set(res["mean"]) >= {"acc", "se", "sp", "f1", "auc", "mcc"}
    assert (tmp_path / "run" / "best.pt").exists()
    assert len(json.loads((tmp_path / "run" / "history.json").read_text())) == 2

    from catnet.evaluate import load_model

    model, ckpt = load_model(str(tmp_path / "run" / "best.pt"), "cpu")
    assert ckpt["train_cfg"]["model"] == "catnet-s"
