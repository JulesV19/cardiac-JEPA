"""Pertes JEPA + régularisateurs anti-collapse VICReg.

L = L_jepa + λ_var * L_var + λ_cov * L_cov

- L_jepa : smooth-L1 entre prédictions et cibles EMA (LayerNormées, stop-grad).
- L_var  : hinge sur l'écart-type par dimension (force std ≥ 1) — empêche l'effondrement
           de la variance des embeddings (collapse).
- L_cov  : pénalise les covariances hors-diagonale — décorrèle les dimensions.

Numérique : ces termes sont TOUJOURS calculés en float32, autocast désactivé. En fp16,
`z.T @ z` sur N≈60 000 lignes produit des entrées ≈ N (max fp16 = 65 504) et déborde en
`inf` -> `NaN`. On divise donc par sqrt(N-1) AVANT le produit : aucun intermédiaire géant.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _fp32(z: torch.Tensor) -> torch.Tensor:
    """Aplatit sur (N, D) et force float32."""
    return z.reshape(-1, z.shape[-1]).float()


def _no_autocast(device_type: str):
    """Désactive l'autocast si actif (sinon les matmul repasseraient en fp16)."""
    if torch.is_autocast_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred, target : (B, n_tgt, D). Cible déjà LayerNormée + détachée en amont."""
    return F.smooth_l1_loss(pred.float(), target.float())


def vicreg_variance(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge variance VICReg. z : (..., D) aplati sur (N, D). Encourage std_dim >= gamma."""
    z = _fp32(z)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """Terme covariance VICReg : somme des carrés hors-diagonale / D. z aplati sur (N, D)."""
    z = _fp32(z)
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    with _no_autocast(z.device.type):
        # cov = zᵀz/(n-1), mais mise à l'échelle avant le produit -> entrées O(1).
        zs = z / (max(n - 1, 1) ** 0.5)
        cov = zs.T @ zs
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
