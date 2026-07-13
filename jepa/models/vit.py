"""Encodeur ViT per-lead : PatchEmbed (projection linéaire par dérivation) + blocs Transformer.

Grille H=12 leads × W=40 patches temporels, token idx = lead*W + time (row-major).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import Block, _PosEmbed, gather_tokens, init_weights


class PatchEmbed(nn.Module):
    """Signal (B, N_SAMPLES, N_LEADS) -> tokens (B, H*W, embed_dim), per-lead."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_len, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_samples, n_leads = x.shape
        H, W, P = self.cfg.grid_h, self.cfg.grid_w, self.cfg.patch_len
        assert n_leads == H and n_samples == W * P, \
            f"attendu ({W*P},{H}), reçu ({n_samples},{n_leads})"
        # (B, T, L) -> (B, L, T) -> (B, L, W, P) -> (B, L*W, P)  [lead-major]
        x = x.transpose(1, 2).reshape(B, H, W, P).reshape(B, H * W, P)
        return self.proj(x)


class Encoder(nn.Module):
    """Encodeur ViT : patch-embed + pos + blocs. Peut ne traiter qu'un sous-ensemble de tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos = _PosEmbed(cfg.grid_h, cfg.grid_w, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [Block(cfg.embed_dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.apply(init_weights)

    def forward(self, signals: torch.Tensor, token_idx: torch.Tensor | None = None):
        tokens = self.patch_embed(signals) + self.pos.table().unsqueeze(0)
        if token_idx is not None:
            tokens = gather_tokens(tokens, token_idx)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)
