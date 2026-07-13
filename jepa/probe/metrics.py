"""Métrique d'évaluation aval : macro-AUROC multi-label (ignore les classes absentes)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..data import SUPERCLASSES


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """AUROC par classe + macro. Ignore les classes absentes du split."""
    per_class = {}
    for j, c in enumerate(SUPERCLASSES):
        if len(np.unique(y_true[:, j])) < 2:      # classe constante -> AUROC indéfinie
            continue
        per_class[c] = roc_auc_score(y_true[:, j], y_score[:, j])
    return float(np.mean(list(per_class.values()))), per_class
