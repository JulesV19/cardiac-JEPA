"""Encodeur xresnet1d branché au JEPA — même interface que ConvEncoder/ViT.

Reprend le *design* xresnet1d (stem 3-convs, downsample ResNet-D = AvgPool + 1×1, dernier
norm de chaque bloc à γ=0) mais remplace **BatchNorm par GroupNorm** : sous input-masking
(zéros injectés dans le contexte) et cible EMA, les stats de batch de la BN sont corrompues.
Le xresnet BN supervisé (`jepa/supervised`) reste la baseline « recette pleine » séparée ;
ici on teste le design xresnet *dans* le JEPA, à interface tokens identique aux autres encodeurs.

Interface : `forward(signals (B, W*P, N_LEADS), token_idx) -> (B, W, embed_dim)`.
Les 12 dérivations sont les CANAUX ; grille de tokens 1×W (H=1). Comme ConvEncoder, un
`AdaptiveAvgPool1d(W)` force exactement W tokens en sortie (le downsample xresnet ne tombe pas
pile sur W ; l'alignement token↔patch est donc approché, propriété partagée avec ConvEncoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import N_LEADS
from .config import ModelConfig
from .layers import gather_tokens, group_norm, init_weights

# arch -> (type de bloc, nb de blocs par étage) — fidèle au benchmark xresnet1d.
_ARCH = {
    "xresnet1d18": ("basic", [2, 2, 2, 2]),
    "xresnet1d34": ("basic", [3, 4, 6, 3]),
    "xresnet1d50": ("bottleneck", [3, 4, 6, 3]),
    "xresnet1d101": ("bottleneck", [3, 4, 23, 3]),
}
_BASE_WIDTHS = [64, 128, 256, 512]


def _conv_gn(in_ch, out_ch, kernel, stride, act, zero_gamma=False):
    """Conv1d(sans biais) + GroupNorm(+GELU). `zero_gamma` : γ init à 0 (dernier norm d'un bloc)."""
    gn = group_norm(out_ch)
    nn.init.zeros_(gn.weight) if zero_gamma else nn.init.ones_(gn.weight)
    layers = [nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False), gn]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _downsample_d(in_ch, out_ch, stride):
    """Skip ResNet-D : AvgPool (anti-aliasing) puis 1×1, plutôt qu'un 1×1 stridé."""
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    down = []
    if stride != 1:
        down.append(nn.AvgPool1d(stride, stride=stride, ceil_mode=True))
    down.append(_conv_gn(in_ch, out_ch, 1, 1, act=False))
    return nn.Sequential(*down)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, mid_ch, stride, kernel):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_gn(in_ch, mid_ch, kernel, stride, act=True),
            _conv_gn(mid_ch, out_ch, kernel, 1, act=False, zero_gamma=True))
        self.down = _downsample_d(in_ch, out_ch, stride)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.down(x) + self.convs(x))


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride, kernel):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.convs = nn.Sequential(
            _conv_gn(in_ch, mid_ch, 1, 1, act=True),
            _conv_gn(mid_ch, mid_ch, kernel, stride, act=True),
            _conv_gn(mid_ch, out_ch, 1, 1, act=False, zero_gamma=True))
        self.down = _downsample_d(in_ch, out_ch, stride)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.down(x) + self.convs(x))


_BLOCKS = {"basic": _BasicBlock, "bottleneck": _Bottleneck}


class XResNetEncoder(nn.Module):
    """xresnet1d (GroupNorm) : (B, W*P, N_LEADS) -> (B, W, embed_dim)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.xr_arch not in _ARCH:
            raise ValueError(f"xr_arch inconnue: {cfg.xr_arch} (dispo: {list(_ARCH)})")
        block_kind, layers = _ARCH[cfg.xr_arch]
        block = _BLOCKS[block_kind]
        k = cfg.xr_kernel
        widths = [max(8, round(w * cfg.xr_width_mult)) for w in _BASE_WIDTHS]

        # Stem xresnet : 3 convs N_LEADS -> w0/2 -> w0/2 -> w0 (1er stride 2) + maxpool.
        s = max(widths[0] // 2, 8)
        stem_szs = [N_LEADS, s, s, widths[0]]
        stem = [_conv_gn(stem_szs[i], stem_szs[i + 1], k, stride=2 if i == 0 else 1, act=True)
                for i in range(3)]
        stem.append(nn.MaxPool1d(3, stride=2, padding=1))

        stages, in_ch = [], stem_szs[-1]
        for i, (mid_ch, n_blocks) in enumerate(zip(widths, layers)):
            stride = 1 if i == 0 else 2          # étage 0 : le maxpool a déjà downsamplé
            for b in range(n_blocks):
                stages.append(block(in_ch, mid_ch, stride if b == 0 else 1, k))
                in_ch = mid_ch * block.expansion
        self.backbone = nn.Sequential(*stem, *stages)

        self.proj = nn.Conv1d(in_ch, cfg.embed_dim, 1)       # -> embed_dim
        self.pool = nn.AdaptiveAvgPool1d(cfg.grid_w)         # force exactement W tokens
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.apply(init_weights)

    def forward(self, signals: torch.Tensor, token_idx: torch.Tensor | None = None):
        B, n_samples, n_leads = signals.shape
        W, P = self.cfg.grid_w, self.cfg.patch_len
        assert n_leads == N_LEADS and n_samples == W * P, \
            f"attendu ({W*P},{N_LEADS}), reçu ({n_samples},{n_leads})"
        x = signals.transpose(1, 2)                          # (B, N_LEADS=canaux, T)
        x = self.pool(self.proj(self.backbone(x)))           # (B, embed_dim, W)
        tokens = x.transpose(1, 2)                           # (B, W, embed_dim)
        if token_idx is not None:
            tokens = gather_tokens(tokens, token_idx)
        return self.norm(tokens)
