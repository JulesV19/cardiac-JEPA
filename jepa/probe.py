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

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from .data import SUPERCLASSES, PTBXLDataset
from .jepa import JEPA
from .models import ModelConfig
from .train import pick_device


@torch.no_grad()
def extract_features(encoder: nn.Module, split: str, device, batch_size: int = 256,
                     workers: int = 2, limit: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Encodeur gelé, signal complet (aucun masquage) -> moyenne des tokens -> (N, 192)."""
    ds = PTBXLDataset(split, with_labels=True, drop_unlabeled=True)
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    encoder.eval()
    feats, ys = [], []
    for x, y in dl:
        z = encoder(x.to(device), None)          # (B, 480, 192)
        feats.append(z.mean(dim=1).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """AUROC par classe + macro. Ignore les classes absentes du split."""
    per_class = {}
    for j, c in enumerate(SUPERCLASSES):
        if len(np.unique(y_true[:, j])) < 2:      # classe constante -> AUROC indéfinie
            continue
        per_class[c] = roc_auc_score(y_true[:, j], y_score[:, j])
    return float(np.mean(list(per_class.values()))), per_class


def train_linear_head(Xtr, ytr, Xva, yva, device, epochs=100, lr=1e-3,
                      weight_decay=1e-4, batch_size=512):
    """Régression logistique multi-label. Sélection du meilleur epoch sur la val."""
    head = nn.Linear(Xtr.shape[1], len(SUPERCLASSES)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    best = (-1.0, None)
    n = len(Xtr_t)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            lossf(head(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            auc, _ = macro_auroc(yva, head(Xva_t).cpu().numpy())
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    return head, best[0]


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


if __name__ == "__main__":
    main()
