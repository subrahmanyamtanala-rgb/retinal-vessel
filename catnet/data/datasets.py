"""Dataset discovery, loading and patch sampling.

Supported layouts (``root`` is the folder you unpacked the dataset into):

``drive``    DRIVE  - ``{training,test}/images/*.tif``, ``1st_manual/*_manual1.gif``, ``mask/*_mask.gif``
             (20 train / 20 test, official split)
``chase``    CHASE_DB1 - ``Image_XXY.jpg`` + ``Image_XXY_1stHO.png`` in one folder
             (first 20 train / last 8 test; FOV estimated)
``stare``    STARE - ``stare-images/im*.ppm`` (or ``*.ppm`` in root) + ``labels-ah/im*.ah.ppm``
             (first 10 train / last 10 test by default; FOV estimated)
``hrf``      HRF - ``images/*.jpg``, ``manual1/*.tif``, ``mask/*_mask.tif``
             (first 5 of each category train / rest test)
``folder``   Generic - ``{train,test}/{images,labels,masks}/<same file name>``; masks optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import estimate_fov, preprocess

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".ppm", ".bmp"}


@dataclass
class Sample:
    name: str
    image: Path
    label: Path
    fov: Path | None = None
    fov_threshold: float = 0.08  # used only when no FOV mask file is given


def _read(path: Path, mode: str | None = None) -> np.ndarray:
    im = Image.open(path)
    if mode:
        im = im.convert(mode)
    return np.asarray(im)


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def _split(items: list, split: str, n_train: int) -> list:
    return items[:n_train] if split == "train" else items[n_train:]


def list_samples(dataset: str, root: str | Path, split: str = "train", **kw) -> list[Sample]:
    root = Path(root)
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    dataset = dataset.lower()

    if dataset == "drive":
        d = root / ("training" if split == "train" else "test")
        out = []
        for img in _files(d / "images"):
            idx = img.stem.split("_")[0]
            lab = next(iter(sorted((d / "1st_manual").glob(f"{idx}_manual1.*"))), None)
            fov = next(iter(sorted((d / "mask").glob(f"{idx}_*mask.*"))), None)
            if lab is None:
                continue  # some DRIVE test releases omit labels
            out.append(Sample(idx, img, lab, fov))
        return out

    if dataset in ("chase", "chase_db1", "chasedb1"):
        imgs = sorted(p for p in root.rglob("Image_*.jpg"))
        out = [Sample(p.stem, p, p.with_name(p.stem + "_1stHO.png")) for p in imgs
               if p.with_name(p.stem + "_1stHO.png").exists()]
        return _split(out, split, kw.get("n_train", 20))

    if dataset == "stare":
        img_dir = root / "stare-images" if (root / "stare-images").is_dir() else root
        lab_dir = root / "labels-ah"
        out = []
        for p in sorted(img_dir.glob("im*.ppm")):
            lab = lab_dir / f"{p.stem}.ah.ppm"
            if lab.exists():
                # STARE's dark border is noisier than CHASE_DB1's; 0.08 leaks into it on some images
                out.append(Sample(p.stem, p, lab, fov_threshold=0.18))
        return _split(out, split, kw.get("n_train", 10))

    if dataset == "hrf":
        imgs = _files(root / "images")
        by_cat: dict[str, list[Sample]] = {}
        for p in imgs:
            lab = next(iter(sorted((root / "manual1").glob(p.stem + ".*"))), None)
            fov = next(iter(sorted((root / "mask").glob(p.stem + "_mask.*"))), None)
            if lab is not None:
                by_cat.setdefault(p.stem.split("_")[-1], []).append(Sample(p.stem, p, lab, fov))
        n = kw.get("n_train_per_cat", 5)
        return [s for cat in sorted(by_cat) for s in _split(by_cat[cat], split, n)]

    if dataset == "folder":
        d = root / split
        out = []
        for img in _files(d / "images"):
            lab = next(iter(sorted((d / "labels").glob(img.stem + ".*"))), None)
            fov = next(iter(sorted((d / "masks").glob(img.stem + ".*"))), None)
            if lab is not None:
                out.append(Sample(img.stem, img, lab, fov))
        return out

    raise KeyError(f"unknown dataset {dataset!r}")


def load_sample(s: Sample, scale: float = 1.0, **prep) -> dict[str, np.ndarray]:
    """Read one sample -> dict(image (H,W) float32 [0,1], label (H,W) {0,1}, fov (H,W) {0,1}, rgb)."""
    rgb = _read(s.image, "RGB")
    lab = _read(s.label, "L").astype(np.float32)
    lab = (lab > (127 if lab.max() > 1 else 0.5)).astype(np.float32)
    fov = (_read(s.fov, "L") > 127).astype(np.float32) if s.fov is not None else estimate_fov(rgb, s.fov_threshold)
    if scale != 1.0:
        h, w = rgb.shape[:2]
        size = (int(round(w * scale)), int(round(h * scale)))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC))
        lab = (np.asarray(Image.fromarray((lab * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) > 127).astype(np.float32)
        fov = (np.asarray(Image.fromarray((fov * 255).astype(np.uint8)).resize(size, Image.NEAREST)) > 127).astype(np.float32)
    img = preprocess(rgb, fov, **prep)
    return {"name": s.name, "image": img, "label": lab * fov, "fov": fov, "rgb": rgb}


# ------------------------------------------------------------------ torch datasets
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allows metrics/preprocessing without torch
    torch, Dataset = None, object


def augment(img: np.ndarray, lab: np.ndarray, fov: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(4))
    img, lab, fov = np.rot90(img, k), np.rot90(lab, k), np.rot90(fov, k)
    if rng.random() < 0.5:
        img, lab, fov = img[:, ::-1], lab[:, ::-1], fov[:, ::-1]
    if rng.random() < 0.8:  # photometric jitter
        img = np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
        img = np.power(img, rng.uniform(0.7, 1.4))
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab), np.ascontiguousarray(fov)


class PatchDataset(Dataset):
    """Random ``patch x patch`` crops centred inside the FOV, with augmentation.

    ``vessel_bias`` is the probability that a crop is centred on a vessel pixel,
    which oversamples thin-vessel regions.
    """

    def __init__(self, items: list[dict], patch: int = 128, samples_per_epoch: int = 2000,
                 augment: bool = True, vessel_bias: float = 0.5, seed: int = 0):
        self.items, self.patch, self.n = items, patch, samples_per_epoch
        self.augment, self.vessel_bias, self.seed = augment, vessel_bias, seed
        self.epoch = 0
        self._fov_idx = [np.argwhere(it["fov"] > 0.5) for it in items]
        self._ves_idx = [np.argwhere(it["label"] > 0.5) for it in items]

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        k = int(rng.integers(len(self.items)))
        it = self.items[k]
        pool = self._ves_idx[k] if (rng.random() < self.vessel_bias and len(self._ves_idx[k])) else self._fov_idx[k]
        cy, cx = pool[int(rng.integers(len(pool)))]
        p = self.patch
        H, W = it["image"].shape
        pad = p // 2
        img = np.pad(it["image"], pad, mode="reflect")
        lab = np.pad(it["label"], pad)
        fov = np.pad(it["fov"], pad)
        y0 = int(np.clip(cy, 0, H - 1))
        x0 = int(np.clip(cx, 0, W - 1))
        sl = (slice(y0, y0 + p), slice(x0, x0 + p))
        img, lab, fov = img[sl], lab[sl], fov[sl]
        if self.augment:
            img, lab, fov = augment(img, lab, fov, rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
        return t(img), t(lab), t(fov)


def load_split(dataset: str, root, split: str, scale: float = 1.0, limit: int | None = None, **kw) -> list[dict]:
    samples = list_samples(dataset, root, split, **kw)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"no {split} samples found for {dataset!r} under {root}")
    return [load_sample(s, scale) for s in samples]


__all__ = ["PatchDataset", "Sample", "augment", "list_samples", "load_sample", "load_split"]
