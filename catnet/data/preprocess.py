"""Fundus image preprocessing.

Standard pipeline used by most vessel-segmentation work:
green channel (highest vessel/background contrast) -> per-image standardisation
inside the FOV -> CLAHE -> gamma correction -> scale to [0, 1].
"""

from __future__ import annotations

import numpy as np


def to_gray(img: np.ndarray, mode: str = "green") -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return img
    if mode == "green":
        return img[..., 1]
    if mode == "luma":
        return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    raise ValueError(mode)


def estimate_fov(img: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """FOV mask for datasets that do not ship one (CHASE_DB1, STARE)."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., 0]  # red channel is brightest inside the FOV
    x = x / max(float(x.max()), 1e-6)
    m = x > threshold
    try:  # clean up with a small opening/closing if SciPy is available
        from scipy import ndimage as ndi

        m = ndi.binary_opening(m, iterations=2)
        m = ndi.binary_fill_holes(m)
        lab, n = ndi.label(m)
        if n > 1:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
    except Exception:  # pragma: no cover
        pass
    return m.astype(np.float32)


def _clahe(x: np.ndarray, clip: float, tiles: int) -> np.ndarray:
    try:
        from skimage.exposure import equalize_adapthist

        return equalize_adapthist(x, kernel_size=max(x.shape) // tiles, clip_limit=clip / 100.0).astype(np.float32)
    except Exception:
        pass
    try:
        import cv2

        c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        return c.apply((x * 255).astype(np.uint8)).astype(np.float32) / 255.0
    except Exception:
        pass
    # fallback: global histogram equalisation
    hist, bins = np.histogram(x.ravel(), 256, (0.0, 1.0))
    cdf = hist.cumsum().astype(np.float64)
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1.0)
    return np.interp(x.ravel(), bins[:-1], cdf).reshape(x.shape).astype(np.float32)


def preprocess(img: np.ndarray, fov: np.ndarray | None = None, channel: str = "green",
               clahe: bool = True, clip: float = 2.0, tiles: int = 8, gamma: float = 1.2) -> np.ndarray:
    """Return a ``float32`` image in [0, 1], shape ``(H, W)``; background outside FOV = 0."""
    g = to_gray(img, channel)
    m = np.ones_like(g, dtype=bool) if fov is None else np.asarray(fov) > 0.5
    vals = g[m]
    g = (g - vals.mean()) / (vals.std() + 1e-6)
    lo, hi = np.percentile(g[m], [0.5, 99.5])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if clahe:
        g = _clahe(g, clip, tiles)
    if gamma and gamma != 1.0:
        g = np.power(g, 1.0 / gamma)
    g[~m] = 0.0
    return g.astype(np.float32)
