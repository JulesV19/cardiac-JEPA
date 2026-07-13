"""Étape 3 du décodage : évaluer sur les zones MASQUÉES — D(z_tgt) vs D(pred).

D(z_tgt) = borne haute (ce qu'un embedding parfait permet de décoder),
D(pred)  = ce que la prédiction du JEPA préserve réellement.
L'écart entre les deux mesure la qualité de prédiction, décontaminée du décodeur.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import PTBXLDataset
from ..masking import MaskCollator
from .reader import ln, to_patches


@torch.no_grad()
def eval_masked(dec, model, device, batch_size, workers, mask_cfg, seed, use_amp=False,
                plots: int = 0, out_dir: Path | None = None):
    """MSE de reconstruction sur les patches masqués : borne haute (z_tgt) vs prédiction (pred)."""
    cfg = model.cfg
    H, W, P = cfg.grid_h, cfg.grid_w, cfg.patch_len
    collator = MaskCollator(mask_cfg, seed=seed)
    dl = DataLoader(PTBXLDataset("test"), batch_size=batch_size, shuffle=False,
                    num_workers=workers, collate_fn=collator)
    dec.eval()
    # pred_ln = D(ln(pred)) : contrôle du bémol méthodo. D a été entraîné sur z_tgt LayerNormé,
    # or pred sort du predictor non-normalisé. On LN pred comme les cibles pour décontaminer la
    # chute d'un éventuel décalage de distribution. Écart pred_ln↔pred = part imputable à l'échelle ;
    # écart pred_ln↔tgt = vraie erreur de prédiction, à distribution appariée.
    agg = {"pred": [0.0, 0], "pred_ln": [0.0, 0], "tgt": [0.0, 0]}   # [sse, n]
    plotted = 0
    for batch in dl:
        x = batch["signals"].to(device)
        cidx = batch["context_idx"].to(device)
        tidx = batch["target_idx"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            z_ctx = model.encoder(x, cidx)                  # (B, n_ctx, D)
            pred = model.predictor(z_ctx, cidx, tidx)       # (B, n_tgt, D) — prédiction JEPA
            z_full = ln(model.target_encoder(x, None))      # (B, H*W, D)
        pred = pred.float()                                 # retour fp32 pour décodeur/MSE/plot
        pred_ln = ln(pred)                                  # même normalisation par-token que z_tgt
        z_tgt = z_full.float().index_select(1, tidx)        # (B, n_tgt, D) — vrais embeddings

        patches = to_patches(x, H, W, P).index_select(1, tidx)   # (B, n_tgt, P) — cible
        rec_pred = dec(pred)
        rec_pred_ln = dec(pred_ln)
        rec_tgt = dec(z_tgt)

        for key, rec in (("pred", rec_pred), ("pred_ln", rec_pred_ln), ("tgt", rec_tgt)):
            agg[key][0] += F.mse_loss(rec, patches, reduction="sum").item()
            agg[key][1] += patches.numel()

        if plots and plotted < plots and out_dir is not None:
            _plot_example(x[0], cidx, tidx, rec_pred[0], rec_tgt[0], cfg,
                          out_dir / f"recon_{plotted}.png")
            plotted += 1

    res = {}
    for key in ("tgt", "pred", "pred_ln"):
        mse = agg[key][0] / agg[key][1]
        res[key] = {"mse": mse, "r2": 1.0 - mse}   # Var≈1 (signal z-normé) -> R² = 1 - MSE
    return res


def _plot_example(x, cidx, tidx, rec_pred, rec_tgt, cfg, path: Path):
    """Trace quelques dérivations : vrai vs reconstruit, zones masquées ombrées."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    H, W, P = cfg.grid_h, cfg.grid_w, cfg.patch_len
    true = x.detach().cpu().numpy()                          # (W*P, H)
    tset = set(tidx.cpu().tolist())
    tpos = {int(t): i for i, t in enumerate(tidx.cpu().tolist())}
    rp = rec_pred.detach().cpu().numpy()                     # (n_tgt, P) — D(pred), prédiction
    rt = rec_tgt.detach().cpu().numpy()                      # (n_tgt, P) — D(z_tgt), borne haute

    leads = [0, 1, 6, 7]                                     # I, II, V1, V2
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 2.0 * len(leads)), sharex=True)
    t = np.arange(W * P)
    for ax, lead in zip(axes, leads):
        ax.plot(t, true[:, lead], color="black", lw=0.8)
        for w in range(W):
            tok = lead * W + w
            if tok in tset:
                sl = slice(w * P, (w + 1) * P)
                ax.plot(t[sl], rt[tpos[tok]], color="royalblue", lw=1.1)   # borne haute D(z_tgt)
                ax.plot(t[sl], rp[tpos[tok]], color="crimson", lw=1.1)     # prédiction D(pred)
                ax.axvspan(w * P, (w + 1) * P, color="crimson", alpha=0.07)
        ax.set_ylabel(f"lead {lead}")
    handles = [Line2D([0], [0], color="black", lw=0.8, label="vrai"),
               Line2D([0], [0], color="royalblue", lw=1.1, label="D(z_tgt) borne haute"),
               Line2D([0], [0], color="crimson", lw=1.1, label="D(pred) prédiction")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes[0].set_title("noir = vrai · bleu = borne haute D(z_tgt) · rouge = prédiction JEPA (zones masquées)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
