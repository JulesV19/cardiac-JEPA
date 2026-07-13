"""Extraction de features : encodeur gelé -> moyenne des tokens -> (N, embed_dim)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..data import PTBXLDataset


@torch.no_grad()
def extract_features(encoder: nn.Module, split: str, device, batch_size: int = 256,
                     workers: int = 2, limit: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Encodeur gelé, signal complet (aucun masquage) -> moyenne des tokens -> (N, 192)."""
    ds = PTBXLDataset(split, with_labels=True, drop_unlabeled=True)
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    encoder.eval()
    feats, ys = [], []
    for x, y in dl:
        z = encoder(x.to(device), None)          # (B, 480, 192)
        feats.append(z.mean(dim=1).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


@torch.no_grad()
def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne / écart-type par feature (préproc linéaire ajusté sur le train, sans fuite)."""
    return X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
