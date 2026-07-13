"""Décodeur de signal sur JEPA gelé — « le JEPA récupère-t-il le tracé sous le masque ? »

Protocole B (lecteur neutre + transfert), pour isoler la qualité de PRÉDICTION du JEPA :
  1. on entraîne le décodeur D sur les VRAIS embeddings-cibles (encodeur EMA sur signal
     complet, LayerNormés — le même z_tgt que jepa.py), sur TOUS les tokens ;
  2. on GÈLE D ;
  3. à l'évaluation, avec masquage : D(z_tgt) = borne haute, D(pred) = prédiction JEPA.

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

import torch

from ..device import pick_device
from ..jepa import JEPA
from ..masking import MaskConfig
from ..models import ModelConfig
from .evaluate import eval_masked
from .reader import PatchDecoder, train_reader


def build_model(args) -> tuple[JEPA, str]:
    """Construit le JEPA (pré-entraîné ou aléatoire iso-architecture) et un tag descriptif."""
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
    return model, tag


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

    model, tag = build_model(args)
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
