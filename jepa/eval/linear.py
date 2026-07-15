"""Tête linéaire (régression logistique multi-label) + sonde rapide de sélection d'epoch."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import SUPERCLASSES
from .features import extract_features, standardize_fit
from .metrics import macro_auroc


def train_linear_head(Xtr, ytr, Xva, yva, device, epochs=100, lr=1e-3,
                      weight_decay=1e-4, batch_size=512):
    """Régression logistique multi-label. Sélection du meilleur epoch sur la val."""
    head = nn.Linear(Xtr.shape[1], len(SUPERCLASSES)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    best = (-1.0, None)
    n = len(Xtr_t)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            lossf(head(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            auc, _ = macro_auroc(yva, head(Xva_t).cpu().numpy())
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    return head, best[0]


def quick_probe_auroc(encoder: nn.Module, device, train_limit: int = 4000,
                      workers: int = 0) -> float:
    """Sonde linéaire *rapide* pour la sélection du meilleur epoch pendant le pré-entraînement.

    Encodeur gelé -> features (folds 1-8 sous-échantillonnés) -> tête logistique -> macro-AUROC
    sur le fold 9 (val). Le fold 10 (test) n'est JAMAIS touché ici : aucune fuite.
    """
    Xtr, ytr = extract_features(encoder, "pretrain", device, workers=workers, limit=train_limit)
    Xva, yva = extract_features(encoder, "val", device, workers=workers)
    mu, sd = standardize_fit(Xtr)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    _, val_auc = train_linear_head(Xtr, ytr, Xva, yva, device)
    return val_auc
