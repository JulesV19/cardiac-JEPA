"""Étape 1 du décodage : lecteur neutre D (embedding -> forme d'onde) et son entraînement.

Décodeur = MLP par-token : Linear(embed_dim->hidden) -> GELU -> Linear(hidden->patch_len).
Le predictor a déjà fait tout le raisonnement inter-tokens ; D n'a qu'à inverser
embedding -> forme d'onde. Entraîné sur les VRAIS embeddings-cibles (encodeur EMA sur signal
complet, LayerNormés), sélection du meilleur epoch sur le fold 9 (MSE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import PTBXLDataset

try:
    from tqdm.auto import tqdm
except ImportError:                       # fallback : pas de barre, aucun crash
    def tqdm(x, **kw):
        return x


def to_patches(x: torch.Tensor, H: int, W: int, P: int) -> torch.Tensor:
    """Signal (B, W*P, H) -> patches par token (B, H*W, P), même ordre que PatchEmbed.

    token idx = lead*W + time (lead-major), patch = P échantillons d'une seule dérivation.
    """
    B = x.shape[0]
    return x.transpose(1, 2).reshape(B, H, W, P).reshape(B, H * W, P)


def ln(z: torch.Tensor) -> torch.Tensor:
    """LayerNorm par token (sans affine) — identique aux cibles de jepa.py."""
    return F.layer_norm(z, (z.shape[-1],))


class PatchDecoder(nn.Module):
    """MLP par-token : embedding (…, D) -> patch signal (…, patch_len)."""

    def __init__(self, embed_dim: int, patch_len: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, patch_len))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@torch.no_grad()
def precompute_tokens(encoder, split, device, batch_size, workers, use_amp):
    """Forward de l'encodeur GELÉ, UNE seule fois -> (embeddings LN, patches cibles).

    Les deux sont aplatis par token en (N*H*W, .) et stockés en **fp16 sur CPU** : l'encodeur
    ne changeant jamais, on ne recalcule pas ce forward coûteux à chaque epoch du lecteur.
    """
    H, W, P = encoder.cfg.grid_h, encoder.cfg.grid_w, encoder.cfg.patch_len
    D = encoder.cfg.embed_dim
    ds = PTBXLDataset(split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    N = len(ds) * H * W
    Z = torch.empty(N, D, dtype=torch.float16)          # préalloué -> pas de pic x2 au concat
    T = torch.empty(N, P, dtype=torch.float16)
    encoder.eval()
    i = 0
    for x in tqdm(dl, desc=f"précalcul {split}", unit="batch"):
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            z = ln(encoder(x, None))                     # (B, H*W, D)
        n = z.shape[0] * H * W
        Z[i:i + n] = z.reshape(-1, D).half().cpu()
        T[i:i + n] = to_patches(x, H, W, P).reshape(-1, P).half().cpu()
        i += n
    return Z, T


def train_reader(dec, encoder, device, epochs, lr, weight_decay, batch_size, workers, use_amp):
    """Entraîne D sur folds 1-8 (cache), sélectionne le meilleur epoch sur fold 9 (MSE)."""
    Ztr, Ttr = precompute_tokens(encoder, "pretrain", device, batch_size, workers, use_amp)
    Zva, Tva = precompute_tokens(encoder, "val", device, batch_size, workers, use_amp)
    gb = (Ztr.nelement() + Ttr.nelement()) * Ztr.element_size() / 1e9
    print(f"  cache tokens : train {tuple(Ztr.shape)}  val {tuple(Zva.shape)}  (~{gb:.1f} Go fp16)")

    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=weight_decay)
    row_bs = 8192                        # batch de TOKENS (le MLP est par-token, indépendant)
    n = Ztr.shape[0]

    def val_mse() -> float:
        dec.eval()
        sse, cnt = 0.0, 0
        with torch.no_grad():
            for j in range(0, Zva.shape[0], row_bs):
                z = Zva[j:j + row_bs].to(device).float()
                t = Tva[j:j + row_bs].to(device).float()
                sse += F.mse_loss(dec(z), t, reduction="sum").item()
                cnt += t.numel()
        return sse / cnt

    best = (float("inf"), None)
    pbar = tqdm(range(epochs), desc="entraînement lecteur", unit="epoch")
    for ep in pbar:
        dec.train()
        perm = torch.randperm(n)
        for j in range(0, n, row_bs):
            idx = perm[j:j + row_bs]
            z = Ztr[idx].to(device).float()
            t = Ttr[idx].to(device).float()
            opt.zero_grad(set_to_none=True)
            F.mse_loss(dec(z), t).backward()
            opt.step()
        va = val_mse()
        pbar.set_postfix(val_mse=f"{va:.4f}")   # % d'epochs + MSE val courante
        if va < best[0]:
            best = (va, {k: v.detach().clone() for k, v in dec.state_dict().items()})
    dec.load_state_dict(best[1])
    return best[0]
