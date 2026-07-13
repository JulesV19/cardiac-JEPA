"""Baseline supervisée : xresnet1d (BatchNorm + ReLU), variantes de profondeur ET de largeur.

Reproduction pure PyTorch de la famille `xresnet1d` (ecg_ptbxl_benchmarking / fastai) :
stem à 3 convs, downsample ResNet-D (AvgPool + 1×1), dernier BN de chaque bloc initialisé à γ=0
(tweak xresnet). Les 12 dérivations sont les CANAUX d'entrée.

Variantes fidèles au benchmark : `xresnet1d18/34` (blocs *basic*, expansion 1),
`xresnet1d50/101` (blocs *bottleneck*, expansion 4). Un `width_mult` scale toutes les largeurs
pour caler la capacité (ex. matcher tes CNN JEPA à 1,27M / 6,25M sur ~17k échantillons — un
xresnet1d101 plein à 33M mémoriserait le train instantanément et ne serait pas iso-capacité).

À la différence de `ConvEncoder` (GroupNorm, pensé pour l'EMA-target/masquage du JEPA), ce modèle
est *standalone supervisé* : BatchNorm est donc sûr et fidèle au benchmark. Il porte sa propre tête
(`self.head`) — pas d'interface tokens, pas de wrapper JEPA.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import N_LEADS, SUPERCLASSES

# arch -> (type de bloc, nombre de blocs par étage)
_ARCH = {
    "xresnet1d18":  ("basic", [2, 2, 2, 2]),
    "xresnet1d34":  ("basic", [3, 4, 6, 3]),
    "xresnet1d50":  ("bottleneck", [3, 4, 6, 3]),
    "xresnet1d101": ("bottleneck", [3, 4, 23, 3]),
}
_BASE_WIDTHS = [64, 128, 256, 512]


def _conv_bn(in_ch: int, out_ch: int, kernel: int, stride: int, act: bool,
             zero_bn: bool = False) -> nn.Sequential:
    """Conv1d(sans biais) + BatchNorm(+ReLU). `zero_bn` : γ init à 0 (dernier BN d'un bloc)."""
    bn = nn.BatchNorm1d(out_ch)
    nn.init.zeros_(bn.weight) if zero_bn else nn.init.ones_(bn.weight)
    layers = [nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _downsample(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Skip ResNet-D : AvgPool (anti-aliasing) puis 1×1, plutôt qu'un 1×1 stridé."""
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    down = []
    if stride != 1:
        down.append(nn.AvgPool1d(stride, stride=stride, ceil_mode=True))
    down.append(_conv_bn(in_ch, out_ch, 1, 1, act=False))
    return nn.Sequential(*down)


class _BasicBlock1D(nn.Module):
    """Bloc basic (expansion 1) : k×k (stride) -> k×k + skip."""
    expansion = 1

    def __init__(self, in_ch: int, mid_ch: int, stride: int, kernel: int):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_bn(in_ch, mid_ch, kernel, stride, act=True),
            _conv_bn(mid_ch, out_ch, kernel, 1, act=False, zero_bn=True),
        )
        self.down = _downsample(in_ch, out_ch, stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.down(x) + self.convs(x))


class _Bottleneck1D(nn.Module):
    """Bloc bottleneck (expansion 4) : 1×1 réduit -> k×k (stride) -> 1×1 étend + skip."""
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int, kernel: int):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_bn(in_ch, mid_ch, 1, 1, act=True),
            _conv_bn(mid_ch, mid_ch, kernel, stride, act=True),
            _conv_bn(mid_ch, out_ch, 1, 1, act=False, zero_bn=True),   # γ=0 : le bloc part en identité
        )
        self.down = _downsample(in_ch, out_ch, stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.down(x) + self.convs(x))


_BLOCKS = {"basic": _BasicBlock1D, "bottleneck": _Bottleneck1D}


class XResNet1d(nn.Module):
    """xresnet1d supervisé : (B, N_SAMPLES, N_LEADS) -> (B, n_classes)."""

    def __init__(self, block_kind: str, layers: list[int], widths: list[int],
                 kernel: int = 5, n_classes: int = len(SUPERCLASSES)):
        super().__init__()
        block = _BLOCKS[block_kind]
        # Stem xresnet : 3 convs 12->w0/2->w0/2->w0 (1er stride 2) + maxpool.
        s = max(widths[0] // 2, 8)
        stem_szs = [N_LEADS, s, s, widths[0]]
        stem = [_conv_bn(stem_szs[i], stem_szs[i + 1], kernel,
                         stride=2 if i == 0 else 1, act=True) for i in range(3)]
        stem.append(nn.MaxPool1d(3, stride=2, padding=1))

        stages, in_ch = [], stem_szs[-1]
        for i, (mid_ch, n_blocks) in enumerate(zip(widths, layers)):
            stride = 1 if i == 0 else 2          # étage 0 : le maxpool a déjà downsamplé
            for b in range(n_blocks):
                stages.append(block(in_ch, mid_ch, stride if b == 0 else 1, kernel))
                in_ch = mid_ch * block.expansion
        self.backbone = nn.Sequential(*stem, *stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_ch, n_classes))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        # Conv/BN : init par défaut PyTorch (Kaiming pour Conv1d) + γ déjà posé dans _conv_bn.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (1000, N_LEADS), f"attendu (B,1000,{N_LEADS}), reçu {tuple(x.shape)}"
        x = x.transpose(1, 2)                    # (B, N_LEADS=canaux, T)
        return self.head(self.backbone(x))       # (B, n_classes)


def make_xresnet(arch: str = "xresnet1d18", width_mult: float = 1.0,
                 kernel: int = 5) -> XResNet1d:
    """Construit un xresnet de la famille. `arch` dans _ARCH ; `width_mult` scale les largeurs."""
    if arch not in _ARCH:
        raise ValueError(f"arch inconnue: {arch} (dispo: {list(_ARCH)})")
    block_kind, layers = _ARCH[arch]
    widths = [max(8, round(w * width_mult)) for w in _BASE_WIDTHS]
    return XResNet1d(block_kind, layers, widths, kernel=kernel)
