"""Architecture JEPA : PatchEmbed per-lead, encodeur ViT-tiny, predictor.

Conventions de tokens (cohérentes avec masking.py) :
- grille H=12 leads × W=40 patches temporels, token idx = lead*W + time (row-major).
- patch = 25 échantillons d'une seule dérivation → projection linéaire (pas de conv).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    grid_h: int = 12          # leads
    grid_w: int = 40          # patches temporels
    patch_len: int = 25       # échantillons par patch (0.25 s @ 100 Hz)
    embed_dim: int = 192      # ViT-tiny
    depth: int = 12
    heads: int = 3
    mlp_ratio: float = 4.0
    pred_dim: int = 96        # predictor plus étroit (bottleneck)
    pred_depth: int = 6
    pred_heads: int = 3

    @property
    def num_tokens(self) -> int:
        return self.grid_h * self.grid_w


class _PosEmbed(nn.Module):
    """Positional embedding factorisé : pos[lead,time] = lead_emb[lead] + time_emb[time]."""

    def __init__(self, grid_h: int, grid_w: int, dim: int):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.lead_emb = nn.Parameter(torch.zeros(grid_h, dim))
        self.time_emb = nn.Parameter(torch.zeros(grid_w, dim))
        nn.init.trunc_normal_(self.lead_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)

    def table(self) -> torch.Tensor:
        # (H, W, dim) -> (H*W, dim), row-major lead-major.
        pos = self.lead_emb[:, None, :] + self.time_emb[None, :, :]
        return pos.reshape(self.grid_h * self.grid_w, -1)


class Block(nn.Module):
    """Bloc Transformer pre-norm standard (MHSA + MLP)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Signal (B, N_SAMPLES, N_LEADS) -> tokens (B, H*W, embed_dim), per-lead."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_len, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_samples, n_leads = x.shape
        H, W, P = self.cfg.grid_h, self.cfg.grid_w, self.cfg.patch_len
        assert n_leads == H and n_samples == W * P, \
            f"attendu ({W*P},{H}), reçu ({n_samples},{n_leads})"
        # (B, T, L) -> (B, L, T) -> (B, L, W, P) -> (B, L*W, P)  [lead-major]
        x = x.transpose(1, 2).reshape(B, H, W, P).reshape(B, H * W, P)
        return self.proj(x)


def _gather(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """tokens (B, N, D), idx (n,) partagé sur le batch -> (B, n, D)."""
    return tokens.index_select(1, idx)


class Encoder(nn.Module):
    """Encodeur ViT : patch-embed + pos + blocs. Peut ne traiter qu'un sous-ensemble de tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos = _PosEmbed(cfg.grid_h, cfg.grid_w, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [Block(cfg.embed_dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.apply(_init_weights)

    def forward(self, signals: torch.Tensor, token_idx: torch.Tensor | None = None):
        tokens = self.patch_embed(signals) + self.pos.table().unsqueeze(0)
        if token_idx is not None:
            tokens = _gather(tokens, token_idx)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class Predictor(nn.Module):
    """Prédit les embeddings des tokens cibles depuis les tokens de contexte.

    Passe unique sur [contexte projeté + mask tokens aux positions cibles], puis lecture
    des positions cibles et reprojection vers embed_dim.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_in = nn.Linear(cfg.embed_dim, cfg.pred_dim)
        self.pos = _PosEmbed(cfg.grid_h, cfg.grid_w, cfg.pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(cfg.pred_dim, cfg.pred_heads, cfg.mlp_ratio)
             for _ in range(cfg.pred_depth)])
        self.norm = nn.LayerNorm(cfg.pred_dim)
        self.proj_out = nn.Linear(cfg.pred_dim, cfg.embed_dim)
        self.apply(_init_weights)

    def forward(self, z_ctx: torch.Tensor, context_idx: torch.Tensor,
                target_idx: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        pos = self.pos.table()                                  # (N, pred_dim)
        ctx = self.proj_in(z_ctx) + pos.index_select(0, context_idx).unsqueeze(0)
        n_tgt = target_idx.shape[0]
        tgt = self.mask_token.expand(B, n_tgt, -1) + \
            pos.index_select(0, target_idx).unsqueeze(0)
        seq = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        pred = seq[:, -n_tgt:, :]                               # positions cibles
        return self.proj_out(pred)                              # (B, n_tgt, embed_dim)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_tiny() -> ModelConfig:
    """Config ViT-tiny par défaut (voir plan)."""
    return ModelConfig()
