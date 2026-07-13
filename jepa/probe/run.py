"""Sonde linéaire : mesure la qualité des représentations JEPA (phase 2).

Encodeur cible EMA **gelé** -> moyenne globale des 480 tokens -> 192 features
-> une couche linéaire -> 5 superclasses (multi-label). Métrique : macro-AUROC.

Splits : sonde entraînée sur folds 1-8, sélection du meilleur epoch sur fold 9,
résultat rapporté sur fold 10 (jamais utilisé jusqu'ici). Les 407 ECG sans
superclasse sont exclus (convention benchmark PTB-XL).

Contrôle indispensable : `--random-init` sonde un encodeur NON entraîné. Si le JEPA
ne bat pas nettement cette baseline, il n'a rien appris d'utile.

Usage :
    python -m jepa.probe --ckpt runs/tiny_v1/ckpt_e99.pt
    python -m jepa.probe --random-init            # baseline de contrôle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..device import pick_device
from ..jepa import JEPA
from ..models import ModelConfig
from .features import extract_features
from .linear import train_linear_head
from .metrics import macro_auroc


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
        tag = f"{args.ckpt} (epoch {ck['epoch']}, encodeur {args.encoder})"
    return model, tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint de pré-entraînement")
    ap.add_argument("--random-init", action="store_true",
                    help="baseline de contrôle : encodeur NON entraîné")
    ap.add_argument("--encoder", choices=["target", "online"], default="target")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="sous-ensemble (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="fichier .json de résultats")
    args = ap.parse_args()
    if not args.ckpt and not args.random_init:
        ap.error("donne --ckpt, ou --random-init pour la baseline de contrôle")

    device = torch.device(args.device) if args.device else pick_device()

    model, tag = build_model(args)
    encoder = model.target_encoder if args.encoder == "target" else model.encoder
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)                   # gelé : la sonde ne l'entraîne pas
    print(f"Sonde sur {tag} | device={device}")

    kw = dict(workers=args.workers, limit=args.limit)
    Xtr, ytr = extract_features(encoder, "pretrain", device, **kw)
    Xva, yva = extract_features(encoder, "val", device, **kw)
    Xte, yte = extract_features(encoder, "test", device, **kw)
    print(f"features : train{Xtr.shape} val{Xva.shape} test{Xte.shape}")

    # Standardisation ajustée sur le train uniquement (préproc linéaire, sans fuite).
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = ((X - mu) / sd for X in (Xtr, Xva, Xte))

    head, val_auc = train_linear_head(Xtr, ytr, Xva, yva, device)
    with torch.no_grad():
        scores = head(torch.from_numpy(Xte).to(device)).cpu().numpy()
    test_auc, per_class = macro_auroc(yte, scores)

    print(f"\nmacro-AUROC  val={val_auc:.4f}   TEST={test_auc:.4f}")
    print("par classe (test) :")
    for c, v in per_class.items():
        print(f"  {c:5s} {v:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": tag, "val_macro_auroc": val_auc, "test_macro_auroc": test_auc,
             "test_per_class": per_class}, indent=2))
        print(f"\nrésultats -> {args.out}")
