"""Pré-entraînement JEPA — package.

`python -m jepa.train` exécute `__main__.py`. La logique est répartie :
- `schedule.py`   : lr / momentum / groupes de params.
- `monitor.py`    : évaluation anti-collapse sur la val.
- `checkpoint.py` : sauvegarde / reprise (latest.pt, best.pt).
- `csvlog.py`     : schéma et ouverture de metrics.csv.
- `run.py`        : boucle d'entraînement (`main`).

`pick_device` est ré-exporté depuis `jepa.device` (compat : `from jepa.train import pick_device`).
"""
from __future__ import annotations

from ..device import pick_device
from .run import main

__all__ = ["main", "pick_device"]
