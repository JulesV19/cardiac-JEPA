"""Module JEPA : encodeur online + encodeur cible EMA (stop-grad) + predictor.

Asymétrie I-JEPA (mécanisme anti-collapse principal) :
- l'encodeur *online* et le predictor reçoivent le gradient,
- l'encodeur *cible* est une copie EMA (aucun gradient), qui voit le signal COMPLET,
- les cibles sont LayerNormées (sans affine) puis détachées.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import Encoder, ModelConfig, Predictor


class JEPA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)                       # online
        self.predictor = Predictor(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)  # cible EMA
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """θ_tgt ← m·θ_tgt + (1-m)·θ_online (params ET buffers)."""
        for p_t, p_o in zip(self.target_encoder.parameters(),
                            self.encoder.parameters()):
            p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
        for b_t, b_o in zip(self.target_encoder.buffers(),
                            self.encoder.buffers()):
            b_t.copy_(b_o)

    def forward(self, signals: torch.Tensor, context_idx: torch.Tensor,
                target_idx: torch.Tensor):
        # Branche online : contexte -> predictor -> prédiction des cibles.
        z_ctx = self.encoder(signals, context_idx)               # (B, n_ctx, D)
        pred = self.predictor(z_ctx, context_idx, target_idx)     # (B, n_tgt, D)

        # Branche cible : signal complet, sans gradient, LayerNorm + detach.
        with torch.no_grad():
            z_full = self.target_encoder(signals, None)          # (B, N, D)
            z_tgt = z_full.index_select(1, target_idx)           # (B, n_tgt, D)
            z_tgt = F.layer_norm(z_tgt, (z_tgt.shape[-1],))
        return pred, z_tgt.detach(), z_ctx
