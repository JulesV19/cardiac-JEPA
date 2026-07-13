"""Encodeur CNN 1D : les 12 dérivations sont les CANAUX d'entrée, grille de tokens 1×40.

Même interface que l'encodeur ViT (`forward(signals, token_idx=None)`). À ne pas confondre :
`grid_h=1` n'est que le nb de lignes de la grille de tokens ; les 12 dérivations physiques
sont les canaux du CNN. Le downsample temporel vient du produit des `cnn_strides` ;
`AdaptiveAvgPool1d(W)` garantit exactement W tokens en sortie.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import N_LEADS
from .config import ModelConfig
from .layers import gather_tokens, group_norm, init_weights


class _ResBlock1D(nn.Module):
    """Bloc résiduel 1D pre-activation-libre : (conv-GN-GELU) ×2 + skip (down si besoin)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.gn1 = group_norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.gn2 = group_norm(out_ch)
        self.act = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), group_norm(out_ch))
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(self.down(x) + h)


class ConvEncoder(nn.Module):
    """Encodeur CNN 1D (dérivations = canaux) : (B, W*P, H) -> (B, W, embed_dim)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.cnn_kernel
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, cfg.cnn_channels[0], k, stride=1, padding=k // 2, bias=False),
            group_norm(cfg.cnn_channels[0]), nn.GELU())
        stages, in_ch = [], cfg.cnn_channels[0]
        for out_ch, stride in zip(cfg.cnn_channels, cfg.cnn_strides):
            for b in range(cfg.cnn_blocks_per_stage):
                stages.append(_ResBlock1D(in_ch, out_ch, k, stride if b == 0 else 1))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Conv1d(in_ch, cfg.embed_dim, 1)      # projection finale -> embed_dim
        self.pool = nn.AdaptiveAvgPool1d(cfg.grid_w)        # force exactement W tokens
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.apply(init_weights)

    def forward(self, signals: torch.Tensor, token_idx: torch.Tensor | None = None):
        B, n_samples, n_leads = signals.shape
        W, P = self.cfg.grid_w, self.cfg.patch_len
        assert n_leads == N_LEADS and n_samples == W * P, \
            f"attendu ({W*P},{N_LEADS}), reçu ({n_samples},{n_leads})"
        x = signals.transpose(1, 2)                          # (B, N_LEADS=canaux, T)
        x = self.pool(self.head(self.stages(self.stem(x))))  # (B, embed_dim, W)
        tokens = x.transpose(1, 2)                           # (B, W, embed_dim)
        if token_idx is not None:
            tokens = gather_tokens(tokens, token_idx)
        return self.norm(tokens)
