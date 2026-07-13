"""Phase 2 — classifieur multi-label (fine-tuning) — package.

`python -m jepa.classify` exécute `__main__.py`. Découpage :
- `model.py`    : `ECGClassifier`.
- `labels.py`   : `labels_of` (labels multi-hot sans charger les signaux).
- `schedule.py` : groupes de params + planning de lr.
- `run.py`      : boucle de fine-tuning (`main`).
"""
from __future__ import annotations

from .model import ECGClassifier
from .run import main

__all__ = ["main", "ECGClassifier"]
