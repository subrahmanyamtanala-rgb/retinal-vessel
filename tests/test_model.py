import pytest
import torch

from catnet.models import MODEL_NAMES, CATBlock, GridAttention, WindowAttention, build_model, count_parameters


@pytest.mark.parametrize("h,w", [(64, 64), (48, 80), (37, 53)])
def test_window_and_grid_attention_preserve_shape(h, w):
    x = torch.randn(2, 32, h, w)
    assert WindowAttention(32, 4, 8)(x).shape == x.shape
    assert GridAttention(32, 4, 4)(x).shape == x.shape


def test_window_partition_is_local():
    """Changing a pixel must not affect outputs in other windows."""
    torch.manual_seed(0)
    m = WindowAttention(8, 2, 4).eval()
    x = torch.randn(1, 8, 8, 8)
    y0 = m(x)
    x2 = x.clone()
    x2[..., 0, 0] += 10.0
    y1 = m(x2)
    diff = (y1 - y0).abs().sum(1)[0]
    assert diff[:4, :4].sum() > 0
    assert torch.allclose(diff[4:, :], torch.zeros_like(diff[4:, :]), atol=1e-5)
    assert torch.allclose(diff[:, 4:], torch.zeros_like(diff[:, 4:]), atol=1e-5)


def test_grid_attention_is_global():
    """A grid group connects pixels H/g apart, i.e. across the whole map."""
    torch.manual_seed(0)
    m = GridAttention(8, 2, 4).eval()
    x = torch.randn(1, 8, 16, 16)
    x2 = x.clone()
    x2[..., 0, 0] += 10.0
    diff = (m(x2) - m(x)).abs().sum(1)[0]
    # (0,0) shares a group with (4k, 4l)
    assert diff[12, 12] > 0 and diff[4, 8] > 0
    assert diff[1, 1] < 1e-5


def test_router_weights_are_a_distribution():
    blk = CATBlock(32, heads=4, window=4, grid=4)
    blk(torch.randn(2, 32, 16, 16))
    w = blk.last_routing
    assert w.shape == (2, 3, 16, 16)
    assert torch.allclose(w.sum(1), torch.ones(2, 16, 16), atol=1e-5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_presets_forward(name):
    torch.manual_seed(0)
    if name == "catnet-s":
        kw = {}
    elif name == "unet":
        kw = dict(widths=(8, 16, 16, 32, 32))
    else:
        kw = dict(widths=(8, 16, 16, 32, 32), depths=(1, 1), heads=(2, 2))
    m = build_model(name, **kw)
    m.train()
    outs = m(torch.randn(2, 1, 64, 64))
    outs = outs if isinstance(outs, list) else [outs]
    assert len(outs) == (1 if name == "unet" else 4)
    assert all(o.shape == (2, 1, 64, 64) for o in outs)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 1, 70, 90))  # non-multiple of 16 -> padded internally
    assert y.shape == (1, 1, 70, 90)


def test_backward_and_param_count():
    m = build_model("catnet")
    n = count_parameters(m)
    assert 1e6 < n < 3e7
    outs = m(torch.randn(2, 1, 64, 64))
    sum(o.mean() for o in outs).backward()
    assert m.enc4.blocks[0].router.local[-1].weight.grad is not None


def test_routing_maps_exposed():
    m = build_model("catnet-s").eval()
    with torch.no_grad():
        m(torch.randn(1, 1, 128, 128))
    maps = m.routing_maps()
    assert set(maps) == {"enc4.0", "enc5.0"}
    assert maps["enc4.0"].shape == (1, 3, 16, 16)


def test_unet_baseline_and_checkpoint_roundtrip():
    from catnet.models import model_from_config

    m = build_model("unet", widths=(8, 16, 16, 32, 32))
    y = m(torch.randn(2, 1, 50, 70))
    assert y.shape == (2, 1, 50, 70)
    m2 = model_from_config(m.cfg.to_dict())
    m2.load_state_dict(m.state_dict())
    c = build_model("catnet-s")
    c2 = model_from_config(c.cfg.to_dict())
    c2.load_state_dict(c.state_dict())
