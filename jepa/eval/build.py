"""Construction de l'encodeur à évaluer : pré-entraîné JEPA, ou aléatoire iso-architecture.

La baseline aléatoire est TOUJOURS iso-architecture : même `ModelConfig` que le bras JEPA
(pré-entraîné), poids non chargés. C'est la seule comparaison honnête « ce que le SSL ajoute ».
"""
from __future__ import annotations

import torch

from ..jepa import JEPA
from ..models import ModelConfig


def build_jepa(model_cfg: ModelConfig, ckpt: str | None, random_init: bool,
               encoder: str = "target") -> tuple[JEPA, str]:
    """Retourne (jepa, tag). `random_init` -> poids aléatoires de `model_cfg` (iso-archi)."""
    if random_init:
        jepa = JEPA(model_cfg)
        tag = f"random-init iso-archi ({model_cfg.encoder_type})"
    else:
        if not ckpt:
            raise ValueError("--ckpt requis quand --random-init n'est pas passé")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        jepa = JEPA(ModelConfig(**ck["cfg"]["model"]))
        jepa.load_state_dict(ck["model"])
        tag = f"{ckpt} (epoch {ck.get('epoch', '?')}, encodeur {encoder})"
    return jepa, tag
