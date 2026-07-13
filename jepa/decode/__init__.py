"""Décodeur de signal sur JEPA gelé — package.

`python -m jepa.decode` exécute `__main__.py`. Découpage :
- `reader.py`   : lecteur neutre D (`PatchDecoder`), précalcul des tokens, entraînement.
- `evaluate.py` : évaluation sur zones masquées (`eval_masked`) + tracés.
- `run.py`      : orchestration (`main`).
"""
from __future__ import annotations

from .evaluate import eval_masked
from .reader import PatchDecoder, train_reader
from .run import main

__all__ = ["main", "PatchDecoder", "train_reader", "eval_masked"]
