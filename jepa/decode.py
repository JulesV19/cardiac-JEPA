"""Décodeur de signal sur JEPA gelé — « le JEPA récupère-t-il le tracé sous le masque ? »

Idée : on rejoue le forward de pré-entraînement (encodeur + predictor GELÉS) pour obtenir
`pred` = la prédiction du JEPA sur les patches masqués, puis on entraîne un petit décodeur
qui remonte de l'espace des embeddings vers le signal (25 échantillons par token).

Protocole B (lecteur neutre + transfert), pour isoler la qualité de PRÉDICTION du JEPA :
  1. on entraîne le décodeur D sur les VRAIS embeddings-cibles (encodeur EMA sur signal
     complet, LayerNormés — le même z_tgt que jepa.py), sur TOUS les tokens ;
  2. on GÈLE D ;
  3. à l'évaluation, avec masquage :
        D(z_tgt)  = borne haute (ce qu'un embedding parfait permet de décoder),
        D(pred)   = ce que la prédiction du JEPA préserve réellement.
     L'écart entre les deux mesure la qualité de prédiction, décontaminée du décodeur.

Décodeur = MLP par-token : Linear(192->256) -> GELU -> Linear(256->25). Le predictor a déjà
fait tout le raisonnement inter-tokens ; D n'a qu'à inverser embedding -> forme d'onde.

Splits (comme la sonde) : train folds 1-8, sélection fold 9, rapport fold 10.
Cible : signal z-normé (ce que l'encodeur ingère). MSE, et R² = 1 - MSE/Var (Var≈1 car z-normé).

Usage :
    python -m jepa.decode --ckpt "1st training/ckpt_e79.pt"
    python -m jepa.decode --ckpt runs/tiny_v1/ckpt_e99.pt --plots 4 --out dec.json
    python -m jepa.decode --random-init            # contrôle : JEPA non entraîné
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PTBXLDataset
from .jepa import JEPA
from .masking import MaskCollator, MaskConfig
from .models import ModelConfig
from .train import pick_device

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


# ---------------------------------------------------------------------------
# Étape 1 : entraîner le lecteur neutre D sur les vrais embeddings z_tgt (tous les tokens).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Étape 3 : évaluer sur les zones MASQUÉES — D(z_tgt) vs D(pred).
# ---------------------------------------------------------------------------
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="contrôle : JEPA non entraîné")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", type=int, default=0, help="nb d'exemples PNG à sauver")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="fichier .json de résultats")
    ap.add_argument("--retrain-decoder", action="store_true",
                    help="ignore le décodeur sauvé et le ré-entraîne")
    args = ap.parse_args()
    if not args.ckpt and not args.random_init:
        ap.error("donne --ckpt, ou --random-init pour le contrôle")

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    use_amp = device.type == "cuda"    # forward encodeur en fp16 sur GPU (pas de VICReg ici)

    if args.random_init:
        # iso-architecture : si un --ckpt est fourni, on reprend SA config (mêmes dims que le
        # JEPA testé) sans charger les poids. Sinon, défaut ViT-tiny.
        if args.ckpt:
            cfg_m = torch.load(args.ckpt, map_location="cpu", weights_only=False)["cfg"]["model"]
            model = JEPA(ModelConfig(**cfg_m))
            tag = f"random-init iso-archi de {args.ckpt}"
        else:
            model = JEPA(ModelConfig())
            tag = "random-init (contrôle, ViT-tiny)"
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = JEPA(ModelConfig(**ck["cfg"]["model"]))
        model.load_state_dict(ck["model"])
        tag = f"{args.ckpt} (epoch {ck['epoch']})"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                             # tout le JEPA gelé
    print(f"Décodeur sur {tag} | device={device}")

    cfg = model.cfg
    dec = PatchDecoder(cfg.embed_dim, cfg.patch_len, hidden=args.hidden).to(device)

    out_dir = Path(args.out).parent if args.out else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)   # décodeur + PNG + result.json y sont écrits
    dec_path = out_dir / "decoder.pt"
    # D dépend du checkpoint (encodeur EMA) et des hyper-params du lecteur -> signature de garde.
    sig = {"embed_dim": cfg.embed_dim, "patch_len": cfg.patch_len,
           "hidden": args.hidden, "tag": tag, "seed": args.seed}

    val_mse = None
    if dec_path.exists() and not args.retrain_decoder:
        blob = torch.load(dec_path, map_location=device, weights_only=False)
        if blob.get("sig") == sig:
            dec.load_state_dict(blob["state_dict"])
            val_mse = blob["val_mse"]
            print(f"Étape 1 — lecteur D rechargé depuis {dec_path} "
                  f"(val_mse={val_mse:.4f}, pas de ré-entraînement)")
        else:
            print(f"Étape 1 — {dec_path} présent mais signature différente -> ré-entraînement")

    if val_mse is None:
        print("Étape 1 — entraînement du lecteur neutre D sur les vrais embeddings :")
        val_mse = train_reader(dec, model.target_encoder, device, args.epochs, args.lr,
                               args.weight_decay, args.batch_size, args.workers, use_amp)
        torch.save({"state_dict": dec.state_dict(), "val_mse": val_mse, "sig": sig}, dec_path)
        print(f"  meilleur lecteur D sauvegardé -> {dec_path}")

    print("\nÉtape 3 — évaluation sur les zones masquées (fold 10) :")
    res = eval_masked(dec, model, device, args.batch_size, args.workers,
                      MaskConfig(), args.seed, use_amp, plots=args.plots, out_dir=out_dir)

    print(f"\nreader val_mse = {val_mse:.4f}")
    print(f"  D(z_tgt)   borne haute      : MSE={res['tgt']['mse']:.4f}  R²={res['tgt']['r2']:.4f}")
    print(f"  D(pred)    prédiction       : MSE={res['pred']['mse']:.4f}  R²={res['pred']['r2']:.4f}")
    print(f"  D(ln pred) préd. LN-appariée : MSE={res['pred_ln']['mse']:.4f}  R²={res['pred_ln']['r2']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": tag, "reader_val_mse": val_mse, "masked": res}, indent=2))
        print(f"\nrésultats -> {args.out}")


if __name__ == "__main__":
    main()
