"""Baseline models trained with exactly the same pipeline as CAT-Net."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .catnet import ResConvBlock


@dataclass
class UNetConfig:
    in_channels: int = 1
    num_classes: int = 1
    widths: tuple[int, ...] = (16, 32, 64, 128, 256)
    arch: str = "unet"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["widths"] = list(d["widths"])
        return d


class UNet(nn.Module):
    """Residual U-Net (all-convolutional) with the same encoder depth as CAT-Net."""

    stride = 16

    def __init__(self, cfg: UNetConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or UNetConfig(**kw)
        cfg.widths = tuple(cfg.widths)
        self.cfg = cfg
        w = cfg.widths
        self.enc = nn.ModuleList([ResConvBlock(cfg.in_channels, w[0])] +
                                 [ResConvBlock(w[i - 1], w[i]) for i in range(1, len(w))])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in range(len(w) - 1, 0, -1)])
        self.dec = nn.ModuleList([ResConvBlock(2 * w[i - 1], w[i - 1]) for i in range(len(w) - 1, 0, -1)])
        self.head = nn.Conv2d(w[0], cfg.num_classes, 1)

    def routing_maps(self) -> dict:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w_ = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w_) % self.stride
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)[..., :h, :w_]
