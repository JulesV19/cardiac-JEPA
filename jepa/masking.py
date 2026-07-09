"""Masquage par blocs 2D sur la grille de tokens (leads × temps), à la I-JEPA.

Grille : H = 12 leads (lignes), W = 40 patches temporels (colonnes) → 480 tokens.
Index d'un token (row-major, lead-major) : idx = lead * W + time.

Collator I-JEPA multi-block :
- échantillonne `num_target_blocks` blocs cibles rectangulaires (aire/aspect tirés),
- le contexte est le COMPLÉMENT de l'union des cibles (étanchéité stricte par construction,
  aucun token gaspillé),
- masques PARTAGÉS sur tout le batch (shapes uniformes, comme dans I-JEPA).

Note : I-JEPA tire un « bloc contexte » rectangulaire d'aire 85-100 % puis lui retire les
cibles — sur sa grille carrée cela revient quasiment au complément. Sur notre grille très
anisotrope (12×40) un bloc rectangulaire ne peut pas atteindre 85 % d'aire sans être clampé
en hauteur, ce qui affamait le contexte (21 % au lieu de ~40 %) et jetait 28 % des tokens.
Le complément évite ce piège.

Renvoie les signaux empilés + les indices de contexte et de cibles. Le predictor
prédit l'ensemble des tokens cibles (uniques) en une passe, chacun portant son propre
positional embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import N_LEADS


@dataclass
class MaskConfig:
    grid_h: int = N_LEADS          # 12 leads
    grid_w: int = 40               # 40 patches temporels (1000 / 25)
    num_target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.20)   # fraction d'aire de la grille
    target_aspect: tuple[float, float] = (0.5, 2.0)    # h/w
    min_context_tokens: int = 48    # garde-fou : rejette les tirages où les cibles avalent tout
    max_retries: int = 50


def _sample_block(h_grid: int, w_grid: int, scale_rng, aspect_rng, rng) -> np.ndarray:
    """Renvoie un masque booléen (h_grid, w_grid) True sur un rectangle échantillonné."""
    area = rng.uniform(*scale_rng) * h_grid * w_grid
    aspect = rng.uniform(*aspect_rng)                 # h/w
    h = int(round(np.sqrt(area * aspect)))
    w = int(round(np.sqrt(area / aspect)))
    h = int(np.clip(h, 1, h_grid))
    w = int(np.clip(w, 1, w_grid))
    top = rng.integers(0, h_grid - h + 1)
    left = rng.integers(0, w_grid - w + 1)
    m = np.zeros((h_grid, w_grid), dtype=bool)
    m[top:top + h, left:left + w] = True
    return m


def sample_masks(cfg: MaskConfig, rng: np.random.Generator):
    """Échantillonne (context_idx, target_idx) pour un batch. Indices 1D row-major.

    context_idx : np.ndarray[int]  (tokens visibles = complément des cibles)
    target_idx  : np.ndarray[int]  (union des blocs cibles)
    Les deux partitionnent exactement la grille : aucun token ignoré.
    """
    H, W = cfg.grid_h, cfg.grid_w
    for _ in range(cfg.max_retries):
        target_union = np.zeros((H, W), dtype=bool)
        for _ in range(cfg.num_target_blocks):
            target_union |= _sample_block(H, W, cfg.target_scale, cfg.target_aspect, rng)

        ctx = ~target_union                        # étanchéité par construction
        if ctx.sum() < cfg.min_context_tokens or not target_union.any():
            continue

        ctx_idx = np.flatnonzero(ctx.reshape(-1))
        tgt_idx = np.flatnonzero(target_union.reshape(-1))
        return ctx_idx.astype(np.int64), tgt_idx.astype(np.int64)

    raise RuntimeError("Échec d'échantillonnage de masque valide — assouplis MaskConfig.")


class MaskCollator:
    """Collator DataLoader : empile les signaux et attache un masque partagé par batch."""

    def __init__(self, cfg: MaskConfig | None = None, seed: int | None = None):
        self.cfg = cfg or MaskConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[torch.Tensor]):
        x = torch.stack(batch, dim=0)              # (B, N_SAMPLES, N_LEADS)
        ctx_idx, tgt_idx = sample_masks(self.cfg, self.rng)
        return {
            "signals": x,
            "context_idx": torch.from_numpy(ctx_idx),   # (n_ctx,)
            "target_idx": torch.from_numpy(tgt_idx),    # (n_tgt,)
        }
