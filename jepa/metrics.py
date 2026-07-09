"""Diagnostics anti-collapse : le cœur du critère de succès de la phase 1.

Le collapse se lit sur les embeddings de l'encodeur :
- std moyen par dimension -> 0  : effondrement de la variance (collapse dur).
- rang effectif faible          : embeddings dans un sous-espace dégénéré (collapse mou).

`collapse_report` renvoie un dict scalaire loggable, `is_collapsing` une sentinelle booléenne.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def embedding_std(z: torch.Tensor) -> float:
    """std moyen par dimension des embeddings. z : (..., D) aplati sur (N, D).

    float32 obligatoire : la variance sur N≈60 000 lignes déborde en fp16.
    """
    z = z.reshape(-1, z.shape[-1]).float()
    return z.std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """Rang effectif = exp(entropie des valeurs propres normalisées de la covariance).

    Vaut D si les dims sont également actives ; s'effondre vers 1 en cas de collapse.
    """
    # eigvalsh pas toujours dispo sur MPS -> calcul sur CPU (métrique de logging, coût négligeable).
    z = z.reshape(-1, z.shape[-1]).float().cpu()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eig / (eig.sum() + eps)
    p = p[p > eps]
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def collapse_report(z_context: torch.Tensor, z_target: torch.Tensor,
                    pred: torch.Tensor) -> dict[str, float]:
    """Métriques scalaires de suivi du collapse sur un batch."""
    return {
        "emb_std_ctx": embedding_std(z_context),
        "emb_std_tgt": embedding_std(z_target),
        "pred_std": embedding_std(pred),
        "eff_rank_ctx": effective_rank(z_context),
        "eff_rank_tgt": effective_rank(z_target),
    }


def is_collapsing(report: dict[str, float], std_threshold: float = 0.1,
                  rank_threshold: float = 2.0) -> bool:
    """Sentinelle : True si les embeddings semblent s'effondrer."""
    return (report["emb_std_ctx"] < std_threshold
            or report["eff_rank_ctx"] < rank_threshold)
