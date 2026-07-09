"""Diagnostics anti-collapse : le cœur du critère de succès de la phase 1.

Le collapse se lit sur les embeddings de l'encodeur :
- std moyen par dimension -> 0  : effondrement de la variance (collapse dur).
- rang effectif faible          : embeddings dans un sous-espace dégénéré (collapse mou).

`collapse_report` renvoie un dict scalaire loggable, `is_collapsing` une sentinelle booléenne.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


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
def prediction_quality(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Qualité de prédiction, INDÉPENDANTE de la difficulté des cibles.

    La loss JEPA brute n'est comparable à rien : la cible bouge (encodeur EMA) et s'étale
    sous l'effet de VICReg, donc l'erreur croît mécaniquement à qualité constante.

    r2  : 1 - MSE(pred, tgt) / Var(tgt). Compare au prédicteur trivial « toujours la
          moyenne des cibles ». 0 = pas mieux que l'idiot, 1 = parfait, <0 = pire que l'idiot.
    cos : similarité cosinus moyenne — pointe-t-on dans la bonne direction ?
    """
    p = pred.reshape(-1, pred.shape[-1]).float()
    t = target.reshape(-1, target.shape[-1]).float()
    mse = (p - t).pow(2).mean()
    var = (t - t.mean(dim=0, keepdim=True)).pow(2).mean()
    r2 = 1.0 - mse / (var + 1e-12)
    cos = F.cosine_similarity(p, t, dim=-1).mean()
    return {"r2": r2.item(), "cos": cos.item()}


@torch.no_grad()
def collapse_report(z_context: torch.Tensor, z_target: torch.Tensor,
                    pred: torch.Tensor) -> dict[str, float]:
    """Métriques scalaires de suivi du collapse et de la qualité de prédiction."""
    return {
        "emb_std_ctx": embedding_std(z_context),
        "emb_std_tgt": embedding_std(z_target),
        "pred_std": embedding_std(pred),
        "eff_rank_ctx": effective_rank(z_context),
        "eff_rank_tgt": effective_rank(z_target),
        **prediction_quality(pred, z_target),
    }


def is_collapsing(report: dict[str, float], std_threshold: float = 0.1,
                  rank_threshold: float = 2.0) -> bool:
    """Sentinelle : True si les embeddings semblent s'effondrer."""
    return (report["emb_std_ctx"] < std_threshold
            or report["eff_rank_ctx"] < rank_threshold)
