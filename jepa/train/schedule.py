"""Plannings d'optimisation du pré-entraînement : groupes de params, lr, momentum EMA."""
from __future__ import annotations

import math

import torch


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Pas de weight decay sur les params 1D (norms, biais, pos embeds, mask token)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "pos" in n or "mask_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def momentum_at(step: int, total: int, m0: float, m1: float) -> float:
    return m0 + (m1 - m0) * step / max(total - 1, 1)
