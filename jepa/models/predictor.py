"""Predictor commun : prédit les embeddings cibles depuis les tokens de contexte.

Passe unique sur [contexte projeté + mask tokens aux positions cibles], puis lecture des
positions cibles et reprojection vers embed_dim. Indépendant de l'encodeur (ViT ou CNN).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import Block, _PosEmbed, init_weights


class Predictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.embed_dim, cfg.pred_dim)
        self.pos = _PosEmbed(cfg.grid_h, cfg.grid_w, cfg.pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(cfg.pred_dim, cfg.pred_heads, cfg.mlp_ratio)
             for _ in range(cfg.pred_depth)])
        self.norm = nn.LayerNorm(cfg.pred_dim)
        self.proj_out = nn.Linear(cfg.pred_dim, cfg.embed_dim)
        self.apply(init_weights)

    def forward(self, z_ctx: torch.Tensor, context_idx: torch.Tensor,
                target_idx: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        pos = self.pos.table()                                  # (N, pred_dim)
        ctx = self.proj_in(z_ctx) + pos.index_select(0, context_idx).unsqueeze(0)
        n_tgt = target_idx.shape[0]
        tgt = self.mask_token.expand(B, n_tgt, -1) + \
            pos.index_select(0, target_idx).unsqueeze(0)
        seq = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        pred = seq[:, -n_tgt:, :]                               # positions cibles
        return self.proj_out(pred)                              # (B, n_tgt, embed_dim)
