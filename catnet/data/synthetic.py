"""Synthetic fundus-like images with known vessel trees.

Used for unit tests, CI smoke-training and quick sanity checks when the public
datasets are not available. The generator draws a few recursive branching trees
radiating from an "optic disc", with tapering widths, a circular FOV, a non-uniform
illumination field, a bright disc, dark lesions and sensor noise. It is not a
substitute for real data: numbers measured on it say nothing about clinical
performance.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def _branch(draw, rng, x, y, ang, width, length, depth, scale):
    if depth == 0 or width < 0.6 or length < 4:
        return
    n = int(rng.integers(4, 9))
    seg = length / n
    for _ in range(n):
        ang += rng.normal(0, 0.18)
        nx, ny = x + seg * math.cos(ang), y + seg * math.sin(ang)
        draw.line([(x * scale, y * scale), (nx * scale, ny * scale)], fill=255, width=max(1, int(round(width * scale))))
        r = width * scale / 2
        draw.ellipse([nx * scale - r, ny * scale - r, nx * scale + r, ny * scale + r], fill=255)
        x, y = nx, ny
        if rng.random() < 0.25:
            _branch(draw, rng, x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1),
                    width * 0.7, length * 0.55, depth - 1, scale)
    _branch(draw, rng, x, y, ang + rng.uniform(0.3, 0.7), width * 0.75, length * 0.65, depth - 1, scale)
    _branch(draw, rng, x, y, ang - rng.uniform(0.3, 0.7), width * 0.75, length * 0.65, depth - 1, scale)


def synthetic_fundus(size: int = 256, seed: int | None = None, n_trees: int = 4):
    """Return ``(rgb uint8 (H,W,3), vessel mask float32 (H,W), fov float32 (H,W))``."""
    rng = np.random.default_rng(seed)
    ss = 4  # supersample for anti-aliased ground truth
    canvas = Image.new("L", (size * ss, size * ss), 0)
    draw = ImageDraw.Draw(canvas)
    cx = size * rng.uniform(0.35, 0.65)
    cy = size * rng.uniform(0.4, 0.6)
    for k in range(n_trees):
        ang = 2 * math.pi * k / n_trees + rng.normal(0, 0.3)
        _branch(draw, rng, cx, cy, ang, width=size / 70 * rng.uniform(0.9, 1.4),
                length=size * 0.35, depth=4, scale=ss)
    soft = np.asarray(canvas.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    vessels = (soft > 0.5).astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size / 2, yy - size / 2)
    fov = (r < size * 0.47).astype(np.float32)

    illum = 0.55 + 0.35 * np.exp(-(((xx - size * rng.uniform(0.3, 0.7)) ** 2 + (yy - size * rng.uniform(0.3, 0.7)) ** 2)
                                    / (2 * (size * 0.45) ** 2)))
    disc = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.04) ** 2)))
    bg = illum + 0.4 * disc
    lesions = np.zeros_like(bg)
    for _ in range(int(rng.integers(0, 5))):
        lx, ly = rng.uniform(0, size, 2)
        lesions += 0.25 * np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * rng.uniform(2, 6) ** 2)))
    contrast = rng.uniform(0.25, 0.45)
    green = bg * (1 - contrast * soft) - lesions + rng.normal(0, 0.03, bg.shape)
    red = np.clip(bg * 1.2 - 0.1 * soft, 0, 1)
    blue = np.clip(bg * 0.3, 0, 1)
    rgb = np.stack([red, np.clip(green, 0, 1), blue], -1) * fov[..., None]
    return (rgb * 255).astype(np.uint8), vessels * fov, fov


def write_synthetic_dataset(root, n_train: int = 16, n_test: int = 4, size: int = 256, seed: int = 0) -> None:
    """Write a dataset in the generic ``images/ labels/ masks/`` layout."""
    from pathlib import Path

    root = Path(root)
    for split, n, off in (("train", n_train, 0), ("test", n_test, 10_000)):
        for sub in ("images", "labels", "masks"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, ves, fov = synthetic_fundus(size, seed=seed + off + i)
            name = f"{i:03d}.png"
            Image.fromarray(img).save(root / split / "images" / name)
            Image.fromarray((ves * 255).astype(np.uint8)).save(root / split / "labels" / name)
            Image.fromarray((fov * 255).astype(np.uint8)).save(root / split / "masks" / name)
