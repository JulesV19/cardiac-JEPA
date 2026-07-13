"""Sonde linéaire — package.

`python -m jepa.probe` exécute `__main__.py`. Découpage :
- `features.py` : extraction des features (encodeur gelé -> moyenne des tokens).
- `metrics.py`  : `macro_auroc`.
- `linear.py`   : `train_linear_head`, `quick_probe_auroc` (sonde de sélection d'epoch).
- `run.py`      : évaluation complète (`main`).

API ré-exportée : `macro_auroc` (utilisé par classify), `quick_probe_auroc` (utilisé par train).
"""
from __future__ import annotations

from .features import extract_features
from .linear import quick_probe_auroc, train_linear_head
from .metrics import macro_auroc
from .run import main

__all__ = ["main", "macro_auroc", "quick_probe_auroc", "extract_features",
           "train_linear_head"]
