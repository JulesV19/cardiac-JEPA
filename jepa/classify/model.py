"""Classifieur multi-label : encodeur JEPA -> moyenne des tokens -> tête linéaire."""
from __future__ import annotations

import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    """Encodeur JEPA -> moyenne des tokens -> tête linéaire multi-label."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, None)          # (B, 480, D), aucun masquage
        return self.head(z.mean(dim=1))    # (B, n_classes)
