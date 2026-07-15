"""Couche d'évaluation unifiée (remplace les anciens paquets probe/ + classify/).

Un seul point d'entrée : `python -m jepa.eval --mode {probe,finetune} ...`.
- `probe`    : sonde linéaire sur encodeur GELÉ (qualité des représentations).
- `finetune` : fine-tuning complet AVEC early-stopping sur la val macro-AUROC.

Tout run écrit un `result.json` homogène : macro-AUROC + macro-AUPRC test, par classe,
avec intervalles de confiance bootstrap. Baseline aléatoire toujours iso-architecture.
`quick_probe_auroc` (sélection d'epoch in-loop du pré-entraînement) est ré-exporté ici.
"""
from __future__ import annotations

from .linear import quick_probe_auroc, train_linear_head
from .metrics import macro_auprc, macro_auroc, summarize

__all__ = ["quick_probe_auroc", "train_linear_head", "macro_auroc", "macro_auprc", "summarize"]
