"""Référence supervisée BN : xresnet1d from-scratch, même boucle/early-stopping/stats que finetune.

Hors grille SSL (qui est GroupNorm iso-archi) : c'est la ligne « meilleure recette » du design
xresnet à pleine BatchNorm. Aucun pré-entraînement, aucun ckpt.
"""
from __future__ import annotations

from pathlib import Path

from ..models.xresnet_supervised import make_xresnet
from .finetune import fit_classifier


def run_supervised(model_cfg: dict, cfg: dict, out_dir: Path, seed: int,
                   train_frac: float, device, limit=None) -> dict:
    model = make_xresnet(**model_cfg)
    tag = f"xresnet supervisé BN {model_cfg.get('arch', 'xresnet1d18')} (from-scratch)"
    return fit_classifier(model, tag, "supervised", cfg, out_dir, seed, train_frac, device, limit)
