"""Briques partagées : bloc Transformer, positional embedding factorisé, inits, helpers.

Utilisées par l'encodeur ViT (`vit.py`), l'encodeur CNN (`cnn.py`) et le predictor
(`predictor.py`).
"""
from __future__ import annotations

from math import gcd

import torch
import torch.nn as nn


class _PosEmbed(nn.Module):
    """Positional embedding factorisé : pos[lead,time] = lead_emb[lead] + time_emb[time]."""

    def __init__(self, grid_h: int, grid_w: int, dim: int):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.lead_emb = nn.Parameter(torch.zeros(grid_h, dim))
        self.time_emb = nn.Parameter(torch.zeros(grid_w, dim))
        nn.init.trunc_normal_(self.lead_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)

    def table(self) -> torch.Tensor:
        # (H, W, dim) -> (H*W, dim), row-major lead-major.
        pos = self.lead_emb[:, None, :] + self.time_emb[None, :, :]
        return pos.reshape(self.grid_h * self.grid_w, -1)


class Block(nn.Module):
    """Bloc Transformer pre-norm standard (MHSA + MLP)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """tokens (B, N, D), idx (n,) partagé sur le batch -> (B, n, D)."""
    return tokens.index_select(1, idx)


def group_norm(ch: int) -> nn.GroupNorm:
    """GroupNorm avec un nb de groupes qui divise `ch` (≤ 8). Pas de BatchNorm : stats de
    batch + cible EMA + input-masking = source de bugs silencieux."""
    return nn.GroupNorm(num_groups=max(gcd(ch, 8), 1), num_channels=ch)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
