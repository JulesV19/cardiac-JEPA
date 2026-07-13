"""Extraction des labels multi-hot sans charger les signaux."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def labels_of(ds) -> np.ndarray:
    """Matrice multi-hot (N, 5) d'un PTBXLDataset ou d'un Subset imbriqué.

    Lit `labels` directement : ne charge aucun signal.
    """
    if isinstance(ds, Subset):
        return labels_of(ds.dataset)[np.asarray(ds.indices)]
    return ds.labels[ds.positions]
