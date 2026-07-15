"""Architectures JEPA, une par fichier.

- `config.py`    : `ModelConfig` (hyperparams) + `build_tiny`.
- `layers.py`    : briques partagées (Block Transformer, positional embed, inits).
- `vit.py`       : encodeur ViT per-lead (`Encoder`, `PatchEmbed`) — grille 12×40.
- `cnn.py`       : encodeur CNN 1D (`ConvEncoder`) — dérivations = canaux, grille 1×40.
- `predictor.py` : predictor commun aux deux encodeurs.

Conventions de tokens (cohérentes avec `masking.py`) :
- ViT : grille H=12 leads × W=40 patches temporels, token idx = lead*W + time (row-major),
  patch = 25 échantillons d'une seule dérivation → projection linéaire (pas de conv).
- CNN : grille H=1 × W=40 (40 tokens temporels), token idx = time ; les 12 dérivations
  physiques sont les CANAUX d'entrée du CNN. Le predictor est inchangé.

Ce `__init__` ré-exporte l'API publique : `from jepa.models import Encoder, ...` reste valide.
"""
from __future__ import annotations

from .cnn import ConvEncoder
from .config import ModelConfig, build_tiny
from .layers import Block
from .predictor import Predictor
from .vit import Encoder, PatchEmbed
from .xresnet_encoder import XResNetEncoder

__all__ = [
    "ModelConfig",
    "build_tiny",
    "Encoder",
    "PatchEmbed",
    "ConvEncoder",
    "XResNetEncoder",
    "Predictor",
    "Block",
]
