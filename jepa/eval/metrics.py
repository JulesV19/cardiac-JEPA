"""Métriques d'évaluation aval + intervalles de confiance bootstrap.

macro-AUROC (métrique de rang) ET macro-AUPRC (sensible au déséquilibre, HYP ~12 %),
chacune par classe et macro. Les IC 95 % sont obtenus par bootstrap sur les patients du
test (rééchantillonnage avec remise) — ils quantifient l'incertitude *aléatoire du jeu de
test*, distincte de la variance inter-graines d'entraînement (agrégée séparément).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data import SUPERCLASSES


def _per_class(y_true, y_score, fn):
    """Applique `fn` classe par classe ; ignore les classes constantes (métrique indéfinie)."""
    per = {}
    for j, c in enumerate(SUPERCLASSES):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        per[c] = float(fn(y_true[:, j], y_score[:, j]))
    return per


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """AUROC par classe + macro. Ignore les classes absentes du split."""
    per = _per_class(y_true, y_score, roc_auc_score)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """AUPRC (average precision) par classe + macro. Ignore les classes absentes."""
    per = _per_class(y_true, y_score, average_precision_score)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def _bootstrap(y_true, y_score, macro_fn, n_boot: int, seed: int):
    """IC 95 % percentile de la métrique macro par rééchantillonnage des lignes (patients)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m, _ = macro_fn(y_true[idx], y_score[idx])
        if not np.isnan(m):
            vals.append(m)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000,
              seed: int = 0) -> dict:
    """Résumé complet loggable : macro AUROC/AUPRC + IC 95 % bootstrap + par classe."""
    auroc, auroc_pc = macro_auroc(y_true, y_score)
    auprc, auprc_pc = macro_auprc(y_true, y_score)
    lo_r, hi_r = _bootstrap(y_true, y_score, macro_auroc, n_boot, seed)
    lo_p, hi_p = _bootstrap(y_true, y_score, macro_auprc, n_boot, seed + 1)
    return {
        "macro_auroc": auroc, "auroc_ci95": [lo_r, hi_r], "auroc_per_class": auroc_pc,
        "macro_auprc": auprc, "auprc_ci95": [lo_p, hi_p], "auprc_per_class": auprc_pc,
        "n_test": int(len(y_true)), "n_boot": n_boot,
    }
