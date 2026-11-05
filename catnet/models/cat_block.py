"""Context-Adaptive Transformer (CAT) block.

A CAT block mixes tokens with three complementary branches and lets every pixel
choose how much of each it wants:

* **Local window attention**  - multi-head self-attention inside non-overlapping
  ``w x w`` windows with a learned relative-position bias. Good at capillary-scale
  detail and vessel edges.
* **Global grid attention**    - attention across a sparse ``g x g`` grid that
  spans the whole feature map (each token attends to tokens that are
  ``H/g`` apart). Carries long-range context: vessel continuity across the optic
  disc, lesions or low-contrast gaps.
* **Directional strip mixer** - depth-wise ``1 x k`` and ``k x 1`` convolutions
  with large ``k``. Cheap, orientation-selective, suited to elongated tubular
  structures.

A **context router** predicts a per-pixel soft assignment over the three branches
from a local descriptor (depth-wise conv) plus an image-level descriptor (global
average/std pooling). This is what makes the block *context-adaptive*: thin
vessels in clean regions can lean on local attention while pixels around
pathology or the optic disc can draw on the global branch.

The feed-forward part is a **context-adaptive FFN**: a depth-wise conv FFN whose
hidden channels are FiLM-modulated by image-level statistics, so the block can
adjust to the strong illumination and contrast shifts between fundus cameras and
datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- utils
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an ``(B, C, H, W)`` tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, ph, pw


def global_context(x: torch.Tensor) -> torch.Tensor:
    """Image-level descriptor: channel-wise mean and std, shape ``(B, 2C)``."""
    mu = x.mean(dim=(2, 3))
    sd = x.flatten(2).std(dim=2, unbiased=False)
    return torch.cat([mu, sd], dim=1)


# ----------------------------------------------------------------------- attention
class _MHSA(nn.Module):
    """Multi-head self-attention over groups of ``n`` tokens with a relative bias."""

    def __init__(self, dim: int, heads: int, size: int, attn_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.size = size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        # relative position bias table, as in Swin
        self.rel_bias = nn.Parameter(torch.zeros((2 * size - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (size - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * size - 1) + rel[..., 1], persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (G, N, C), N = size*size
        g, n, c = t.shape
        qkv = self.qkv(t).reshape(g, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = self.rel_bias[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.attn_drop if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(g, n, c))


class WindowAttention(nn.Module):
    """Self-attention inside non-overlapping ``w x w`` windows (local context)."""

    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.w = window
        self.attn = _MHSA(dim, heads, window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w_ = x.shape
        x, ph, pw = _pad_to_multiple(x, self.w)
        hp, wp = x.shape[-2:]
        s = self.w
        t = x.view(b, c, hp // s, s, wp // s, s).permute(0, 2, 4, 3, 5, 1).reshape(-1, s * s, c)
        t = self.attn(t)
        x = t.view(b, hp // s, wp // s, s, s, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        return x[..., :h, :w_]


class GridAttention(nn.Module):
    """Sparse global attention over a ``g x g`` grid that spans the whole map."""

    def __init__(self, dim: int, heads: int, grid: int):
        super().__init__()
        self.g = grid
        self.attn = _MHSA(dim, heads, grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w_ = x.shape
        x, ph, pw = _pad_to_multiple(x, self.g)
        hp, wp = x.shape[-2:]
        g = self.g
        # token (i, j) of group (p, q) sits at (i * hp/g + p, j * wp/g + q)
        t = x.view(b, c, g, hp // g, g, wp // g).permute(0, 3, 5, 2, 4, 1).reshape(-1, g * g, c)
        t = self.attn(t)
        x = t.view(b, hp // g, wp // g, g, g, c).permute(0, 5, 3, 1, 4, 2).reshape(b, c, hp, wp)
        return x[..., :h, :w_]


class StripMixer(nn.Module):
    """Directional token mixer: depth-wise horizontal + vertical strip convolutions."""

    def __init__(self, dim: int, k: int = 11):
        super().__init__()
        self.pre = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim)
        self.v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        return self.proj(self.h(x) + self.v(x))


# -------------------------------------------------------------------------- router
class ContextRouter(nn.Module):
    """Predicts a per-pixel soft assignment over ``n`` token-mixing branches.

    logits = local_head(DWConv(x)) + global_head([mean(x), std(x)])
    weights = softmax(logits / tau)
    """

    def __init__(self, dim: int, n_branches: int = 3, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, n_branches, 1),
        )
        hidden = max(dim // 4, 16)
        self.glob = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, n_branches))
        # start from a uniform mixture so early training is stable
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.glob[-1].weight)
        nn.init.zeros_(self.glob[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.local(x) + self.glob(global_context(x))[:, :, None, None]
        return torch.softmax(logits / self.tau, dim=1)


# ----------------------------------------------------------------------------- FFN
class ContextAdaptiveFFN(nn.Module):
    """Conv FFN whose hidden channels are FiLM-modulated by image-level context."""

    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.film = nn.Linear(2 * dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(global_context(x)).chunk(2, dim=1)
        h = self.act(self.dw(self.fc1(x)))
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.drop(self.fc2(self.drop(h)))


# ----------------------------------------------------------- anti-collapse tools
def drop_one_branch(w: torch.Tensor, p: float) -> torch.Tensor:
    """With probability ``p`` per sample, zero one random branch and renormalise.

    Forces every branch to stay useful, so the router cannot silently switch one off.
    """
    b, n = w.shape[:2]
    hit = torch.rand(b, device=w.device) < p
    if not hit.any():
        return w
    which = torch.randint(0, n, (b,), device=w.device)
    mask = torch.ones(b, n, device=w.device, dtype=w.dtype)
    mask[hit, which[hit]] = 0.0
    w = w * mask[:, :, None, None]
    return w / w.sum(1, keepdim=True).clamp_min(1e-6)


def routing_regulariser(w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Losses on routing weights ``w`` of shape (B, n, H, W).

    * ``balance``: n * sum_i m_i^2 - 1, where m_i is the mean usage of branch i over
      the batch and all pixels. It is 0 for an even split and n - 1 when one branch takes
      everything (Switch-Transformer style load balancing).
    * ``entropy``: mean per-pixel entropy divided by log n, in [0, 1]. Penalising it makes
      each pixel commit to a branch. Combined with ``balance``, the only way to make both
      small is for *different pixels to choose different branches*.
    """
    n = w.shape[1]
    m = w.mean(dim=(0, 2, 3))
    balance = n * (m * m).sum() - 1.0
    ent = -(w * torch.log(w.clamp_min(1e-8))).sum(1).mean() / torch.log(torch.tensor(float(n)))
    return {"balance": balance, "entropy": ent}


def routing_stats(w: torch.Tensor) -> dict[str, float]:
    """Diagnostics: mean usage per branch, spatial std of the weights, and entropies."""
    n = w.shape[1]
    usage = w.mean(dim=(0, 2, 3))
    spatial_std = w.flatten(2).std(dim=2).mean().item()  # 0 => same mix at every pixel
    pix_ent = (-(w * torch.log(w.clamp_min(1e-8))).sum(1).mean() / torch.log(torch.tensor(float(n)))).item()
    marg_ent = (-(usage * torch.log(usage.clamp_min(1e-8))).sum() / torch.log(torch.tensor(float(n)))).item()
    return {"usage": usage.tolist(), "spatial_std": spatial_std, "pixel_entropy": pix_ent, "marginal_entropy": marg_ent}


# --------------------------------------------------------------------------- block
class CATBlock(nn.Module):
    """Context-Adaptive Transformer block.

    Args:
        dim: channels.
        heads: attention heads for the window and grid branches.
        window: local window size.
        grid: global grid size.
        strip_kernel: kernel length of the directional strip mixer.
        mlp_ratio: FFN expansion.
        drop_path: stochastic depth rate.
        branches: subset of ``("local", "global", "strip")`` (for ablations).
        adaptive: if False, branches are averaged with fixed equal weights
            (ablation "w/o context router").
        film: if False, the FFN ignores image-level context (ablation).
    """

    BRANCHES = ("local", "global", "strip")

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        window: int = 8,
        grid: int = 8,
        strip_kernel: int = 11,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        branches: tuple[str, ...] = BRANCHES,
        adaptive: bool = True,
        film: bool = True,
        layer_scale: float = 1e-2,
        branch_drop: float = 0.0,
    ):
        super().__init__()
        unknown = set(branches) - set(self.BRANCHES)
        if unknown or not branches:
            raise ValueError(f"invalid branches {branches!r}")
        self.branch_names = tuple(branches)
        self.norm1 = LayerNorm2d(dim)
        mixers = {
            "local": lambda: WindowAttention(dim, heads, window),
            "global": lambda: GridAttention(dim, heads, grid),
            "strip": lambda: StripMixer(dim, strip_kernel),
        }
        self.mixers = nn.ModuleDict({n: mixers[n]() for n in self.branch_names})
        self.adaptive = adaptive and len(self.branch_names) > 1
        self.router = ContextRouter(dim, len(self.branch_names)) if self.adaptive else None
        self.norm2 = LayerNorm2d(dim)
        self.ffn = ContextAdaptiveFFN(dim, mlp_ratio)
        if not film:
            self.ffn.film.requires_grad_(False)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path)
        self.branch_drop = branch_drop
        self.last_routing: torch.Tensor | None = None  # for visualisation
        self.routing_for_loss: torch.Tensor | None = None  # non-detached, for the router regulariser

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self.mixers[n](x) for n in self.branch_names]
        if len(outs) == 1:
            return outs[0]
        if self.router is None:
            return torch.stack(outs, 0).mean(0)
        w = self.router(x)
        self.last_routing = w.detach()
        self.routing_for_loss = w if self.training else None
        if self.training and self.branch_drop > 0:
            w = drop_one_branch(w, self.branch_drop)
        return sum(w[:, i : i + 1] * o for i, o in enumerate(outs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma1[:, None, None] * self.mix(self.norm1(x)))
        x = x + self.drop_path(self.gamma2[:, None, None] * self.ffn(self.norm2(x)))
        return x


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


__all__ = [
    "CATBlock",
    "ContextAdaptiveFFN",
    "ContextRouter",
    "GridAttention",
    "LayerNorm2d",
    "StripMixer",
    "WindowAttention",
    "count_parameters",
    "drop_one_branch",
    "global_context",
    "routing_regulariser",
    "routing_stats",
]
