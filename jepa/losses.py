"""Pertes JEPA + régularisateurs anti-collapse VICReg.

L = L_jepa + λ_var * L_var + λ_cov * L_cov

- L_jepa : smooth-L1 entre prédictions et cibles EMA (LayerNormées, stop-grad).
- L_var  : hinge sur l'écart-type par dimension (force std ≥ 1) — empêche l'effondrement
           de la variance des embeddings (collapse).
- L_cov  : pénalise les covariances hors-diagonale — décorrèle les dimensions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred, target : (B, n_tgt, D). Cible déjà LayerNormée + détachée en amont."""
    return F.smooth_l1_loss(pred, target)


def vicreg_variance(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge variance VICReg. z : (..., D) aplati sur (N, D). Encourage std_dim >= gamma."""
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """Terme covariance VICReg : somme des carrés hors-diagonale / D. z aplati sur (N, D)."""
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def total_loss(pred, target, z_context, lambda_var: float = 1.0,
               lambda_cov: float = 0.04):
    """Perte totale + dict de composantes (pour le logging).

    z_context : embeddings de l'encodeur (contexte) sur lesquels on régularise la variance.
    """
    l_jepa = jepa_loss(pred, target)
    l_var = vicreg_variance(z_context)
    l_cov = vicreg_covariance(z_context)
    loss = l_jepa + lambda_var * l_var + lambda_cov * l_cov
    return loss, {"jepa": l_jepa.detach(), "var": l_var.detach(),
                  "cov": l_cov.detach(), "total": loss.detach()}
