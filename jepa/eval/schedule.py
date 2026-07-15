"""Groupes de params (lr différencié encodeur/tête) et planning de lr du fine-tuning."""
from __future__ import annotations

import math

import torch.nn as nn


def build_param_groups(model: nn.Module, head_lr: float, encoder_lr: float,
                       weight_decay: float):
    """4 groupes : {encodeur, tête} x {avec, sans weight decay}.

    Pas de weight decay sur les params 1D (norms, biais, pos embeds). `base_lr` mémorisé
    par groupe ; le scheduler applique un multiplicateur commun. La tête est reconnue au
    préfixe `head`, tout le reste est « encodeur ».
    """
    groups = {
        ("enc", True):   {"params": [], "weight_decay": weight_decay, "base_lr": encoder_lr},
        ("enc", False):  {"params": [], "weight_decay": 0.0,          "base_lr": encoder_lr},
        ("head", True):  {"params": [], "weight_decay": weight_decay, "base_lr": head_lr},
        ("head", False): {"params": [], "weight_decay": 0.0,          "base_lr": head_lr},
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if n.startswith("head") else "enc"
        decay = not (p.ndim <= 1 or "pos" in n)
        groups[(part, decay)]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def lr_mult(step: int, total: int, warmup: int) -> float:
    """Multiplicateur de lr : warmup linéaire puis cosine."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * prog))
