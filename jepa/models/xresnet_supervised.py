"""xresnet1d supervisé (BatchNorm+ReLU) — référence « meilleure recette », hors grille SSL.

Reproduction pure PyTorch de la famille xresnet1d (stem 3-convs, downsample ResNet-D,
dernier BN de chaque bloc à γ=0). Porte sa propre tête de classification : ce n'est PAS un
encodeur JEPA (pour ça, voir `xresnet_encoder.py`, GroupNorm). Sert de point de comparaison
« design xresnet à pleine recette », entraîné from-scratch via `python -m jepa.eval --mode supervised`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import N_LEADS, SUPERCLASSES

_ARCH = {
    "xresnet1d18": ("basic", [2, 2, 2, 2]),
    "xresnet1d34": ("basic", [3, 4, 6, 3]),
    "xresnet1d50": ("bottleneck", [3, 4, 6, 3]),
    "xresnet1d101": ("bottleneck", [3, 4, 23, 3]),
}
_BASE_WIDTHS = [64, 128, 256, 512]


def _conv_bn(in_ch, out_ch, kernel, stride, act, zero_bn=False):
    bn = nn.BatchNorm1d(out_ch)
    nn.init.zeros_(bn.weight) if zero_bn else nn.init.ones_(bn.weight)
    layers = [nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _downsample(in_ch, out_ch, stride):
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    down = []
    if stride != 1:
        down.append(nn.AvgPool1d(stride, stride=stride, ceil_mode=True))
    down.append(_conv_bn(in_ch, out_ch, 1, 1, act=False))
    return nn.Sequential(*down)


class _BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, mid_ch, stride, kernel):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_bn(in_ch, mid_ch, kernel, stride, act=True),
            _conv_bn(mid_ch, out_ch, kernel, 1, act=False, zero_bn=True))
        self.down = _downsample(in_ch, out_ch, stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.down(x) + self.convs(x))


class _Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride, kernel):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_bn(in_ch, mid_ch, 1, 1, act=True),
            _conv_bn(mid_ch, mid_ch, kernel, stride, act=True),
            _conv_bn(mid_ch, out_ch, 1, 1, act=False, zero_bn=True))
        self.down = _downsample(in_ch, out_ch, stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.down(x) + self.convs(x))


_BLOCKS = {"basic": _BasicBlock1D, "bottleneck": _Bottleneck1D}


class XResNet1d(nn.Module):
    """xresnet1d supervisé : (B, 1000, N_LEADS) -> (B, n_classes)."""

    def __init__(self, block_kind, layers, widths, kernel=5, n_classes=len(SUPERCLASSES)):
        super().__init__()
        block = _BLOCKS[block_kind]
        s = max(widths[0] // 2, 8)
        stem_szs = [N_LEADS, s, s, widths[0]]
        stem = [_conv_bn(stem_szs[i], stem_szs[i + 1], kernel, stride=2 if i == 0 else 1, act=True)
                for i in range(3)]
        stem.append(nn.MaxPool1d(3, stride=2, padding=1))
        stages, in_ch = [], stem_szs[-1]
        for i, (mid_ch, n_blocks) in enumerate(zip(widths, layers)):
            stride = 1 if i == 0 else 2
            for b in range(n_blocks):
                stages.append(block(in_ch, mid_ch, stride if b == 0 else 1, kernel))
                in_ch = mid_ch * block.expansion
        self.backbone = nn.Sequential(*stem, *stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_ch, n_classes))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        assert x.shape[1:] == (1000, N_LEADS), f"attendu (B,1000,{N_LEADS}), reçu {tuple(x.shape)}"
        x = x.transpose(1, 2)
        return self.head(self.backbone(x))


def make_xresnet(arch="xresnet1d18", width_mult=1.0, kernel=5) -> XResNet1d:
    if arch not in _ARCH:
        raise ValueError(f"arch inconnue: {arch} (dispo: {list(_ARCH)})")
    block_kind, layers = _ARCH[arch]
    widths = [max(8, round(w * width_mult)) for w in _BASE_WIDTHS]
    return XResNet1d(block_kind, layers, widths, kernel=kernel)
