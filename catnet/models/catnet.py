"""CAT-Net: a hybrid CNN / Context-Adaptive Transformer U-shaped network.

    input (B, in_ch, H, W)
      │
      ├─ Stage 1  ConvStage           C1  @ 1/1   ─────────────── skip ─┐
      ├─ Stage 2  ConvStage           C2  @ 1/2   ─────────── skip ─┐   │
      ├─ Stage 3  ConvStage           C3  @ 1/4   ─────── skip ─┐   │   │
      ├─ Stage 4  CAT blocks x d4     C4  @ 1/8   ─── skip ─┐   │   │   │
      └─ Stage 5  CAT blocks x d5     C5  @ 1/16 (bottleneck)│   │   │   │
                                                       up+CAG  up+CAG ...  → 1x1 → logits
                                                       (deep-supervision heads at 1/8, 1/4, 1/2)

Shallow stages stay convolutional: at full resolution the vessels are 1-10 px wide
and convolutions are both cheaper and better at that scale. The transformer stages
start at 1/8, where a 128 px training patch is 16 x 16 tokens.

Skip connections go through a **context-adaptive gate (CAG)**: the decoder feature,
which already carries global context, produces a spatial-and-channel gate that
suppresses background texture (choroidal pattern, optic-disc rim, lesions) in the
encoder feature before fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cat_block import CATBlock, LayerNorm2d


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )


class ResConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.c2(self.drop(self.c1(x))) + self.skip(x))


class ContextAdaptiveGate(nn.Module):
    """Gate an encoder skip ``s`` with the (upsampled) decoder feature ``d``."""

    def __init__(self, c_skip: int, c_dec: int):
        super().__init__()
        mid = max(c_skip // 2, 8)
        self.ws = nn.Conv2d(c_skip, mid, 1)
        self.wd = nn.Conv2d(c_dec, mid, 1)
        self.spatial = nn.Conv2d(mid, 1, 1)
        self.channel = nn.Sequential(nn.Linear(c_dec, mid), nn.GELU(), nn.Linear(mid, c_skip))

    def forward(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.spatial(F.gelu(self.ws(s) + self.wd(d))))
        c = torch.sigmoid(self.channel(d.mean(dim=(2, 3))))[:, :, None, None]
        # residual form: each factor lies in (0.5, 1.5), so the gate can damp or boost
        # the skip but never zero it out completely (keeps gradients flowing early on)
        return s * (0.5 + a) * (0.5 + c)


class DecoderStage(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, gated: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.gate = ContextAdaptiveGate(c_skip, c_out) if gated else None
        self.fuse = ResConvBlock(c_out + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gate is not None:
            skip = self.gate(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class TransformerStage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int, heads: int, window: int, grid: int,
                 drop_path: list[float], **block_kw):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(cin, cout, 2, stride=2), LayerNorm2d(cout))
        self.blocks = nn.Sequential(*[
            CATBlock(cout, heads=heads, window=window, grid=grid, drop_path=drop_path[i], **block_kw)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


@dataclass
class CATNetConfig:
    in_channels: int = 1
    num_classes: int = 1
    widths: tuple[int, int, int, int, int] = (32, 64, 128, 256, 320)
    depths: tuple[int, int] = (2, 2)          # CAT blocks at 1/8 and 1/16
    heads: tuple[int, int] = (4, 8)
    window: int = 8
    grid: int = 4
    strip_kernel: int = 11
    mlp_ratio: float = 4.0
    drop_path: float = 0.1
    conv_dropout: float = 0.1
    branches: tuple[str, ...] = ("local", "global", "strip")
    adaptive_routing: bool = True
    film: bool = True
    gated_skips: bool = True
    deep_supervision: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CATNetConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("widths", "depths", "heads", "branches"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


class CATNet(nn.Module):
    """Context-Adaptive Transformer network for retinal vessel segmentation.

    ``forward`` returns logits of shape ``(B, num_classes, H, W)``; in training mode
    with deep supervision it returns ``[main, aux_1/2, aux_1/4, aux_1/8]`` (all
    upsampled to the input size).
    """

    def __init__(self, cfg: CATNetConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or CATNetConfig(**kw)
        self.cfg = cfg
        c1, c2, c3, c4, c5 = cfg.widths
        self.enc1 = nn.Sequential(ConvBNAct(cfg.in_channels, c1), ResConvBlock(c1, c1, cfg.conv_dropout))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c1, c2, cfg.conv_dropout), ResConvBlock(c2, c2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c2, c3, cfg.conv_dropout), ResConvBlock(c3, c3))

        dpr = torch.linspace(0, cfg.drop_path, sum(cfg.depths)).tolist()
        block_kw = dict(strip_kernel=cfg.strip_kernel, mlp_ratio=cfg.mlp_ratio, branches=cfg.branches,
                        adaptive=cfg.adaptive_routing, film=cfg.film)
        self.enc4 = TransformerStage(c3, c4, cfg.depths[0], cfg.heads[0], cfg.window, cfg.grid,
                                     dpr[: cfg.depths[0]], **block_kw)
        self.enc5 = TransformerStage(c4, c5, cfg.depths[1], cfg.heads[1], cfg.window, cfg.grid,
                                     dpr[cfg.depths[0]:], **block_kw)

        g = cfg.gated_skips
        self.dec4 = DecoderStage(c5, c4, c4, g)
        self.dec3 = DecoderStage(c4, c3, c3, g)
        self.dec2 = DecoderStage(c3, c2, c2, g)
        self.dec1 = DecoderStage(c2, c1, c1, g)
        self.head = nn.Conv2d(c1, cfg.num_classes, 1)
        if cfg.deep_supervision:
            self.aux = nn.ModuleList([nn.Conv2d(c, cfg.num_classes, 1) for c in (c2, c3, c4)])
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @property
    def stride(self) -> int:
        return 16

    def routing_maps(self) -> dict[str, torch.Tensor]:
        """Per-block branch weights from the last forward pass (B, n_branches, h, w)."""
        out = {}
        for sname, stage in (("enc4", self.enc4), ("enc5", self.enc5)):
            for i, blk in enumerate(stage.blocks):
                if blk.last_routing is not None:
                    out[f"{sname}.{i}"] = blk.last_routing
        return out

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.stride, (-w) % self.stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out = self.head(d1)[..., :h, :w]
        if self.training and self.cfg.deep_supervision:
            size = x.shape[-2:]
            aux = [F.interpolate(a(f), size=size, mode="bilinear", align_corners=False)[..., :h, :w]
                   for a, f in zip(self.aux, (d2, d3, d4))]
            return [out, *aux]
        return out


def build_model(name: str = "catnet", **kw) -> nn.Module:
    """Factory used by the training scripts. ``name`` selects a preset."""
    presets = {
        "catnet": {},
        "catnet-s": dict(widths=(16, 32, 64, 128, 192), depths=(1, 1), heads=(2, 4)),
        "catnet-l": dict(widths=(48, 96, 192, 384, 512), depths=(2, 4), heads=(6, 8)),
        # ablations
        "catnet-no-router": dict(adaptive_routing=False),
        "catnet-no-film": dict(film=False),
        "catnet-no-gate": dict(gated_skips=False),
        "catnet-local-only": dict(branches=("local",)),
        "catnet-global-only": dict(branches=("global",)),
        "catnet-no-strip": dict(branches=("local", "global")),
    }
    if name not in presets:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(presets)}")
    cfg = {**presets[name], **kw}
    return CATNet(CATNetConfig(**cfg))


MODEL_NAMES = (
    "catnet", "catnet-s", "catnet-l", "catnet-no-router", "catnet-no-film", "catnet-no-gate",
    "catnet-local-only", "catnet-global-only", "catnet-no-strip",
)
