"""Sonde linéaire sur encodeur GELÉ : mesure la qualité brute des représentations.

Encodeur (target EMA) gelé -> moyenne des tokens -> tête logistique -> 5 superclasses.
Sélection d'epoch de la tête sur fold 9, test sur fold 10 (macro-AUROC + AUPRC + IC bootstrap).
`--random-init` sonde un encodeur non entraîné (iso-archi) : si le JEPA ne le bat pas nettement,
il n'a rien appris.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..models import ModelConfig
from ..progress import tqdm
from .build import build_jepa
from .features import extract_features, standardize_fit
from .linear import train_linear_head
from .metrics import summarize


def run_probe(model_cfg: ModelConfig, ckpt, random_init, encoder, out_dir: Path,
              seed: int, device, workers: int = 0, limit=None) -> dict:
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    jepa, tag = build_jepa(model_cfg, ckpt, random_init, encoder)
    enc = (jepa.target_encoder if encoder == "target" else jepa.encoder).to(device)
    for p in enc.parameters():
        p.requires_grad_(False)                       # gelé

    kw = dict(workers=workers, limit=limit)
    Xtr, ytr = extract_features(enc, "pretrain", device, **kw)
    Xva, yva = extract_features(enc, "val", device, **kw)
    Xte, yte = extract_features(enc, "test", device, **kw)
    mu, sd = standardize_fit(Xtr)
    Xtr, Xva, Xte = ((X - mu) / sd for X in (Xtr, Xva, Xte))

    head, val_auc = train_linear_head(Xtr, ytr, Xva, yva, device)
    with torch.no_grad():
        scores = head(torch.from_numpy(Xte).to(device)).cpu().numpy()
    stats = summarize(yte, scores, seed=seed)

    result = {"tag": tag, "mode": "probe", "seed": seed,
              "val_macro_auroc": val_auc, **stats}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    ci = stats["auroc_ci95"]
    tqdm.write(f"  probe s{seed:<2} {'':18} AUROC {stats['macro_auroc']:.4f} "
               f"[{ci[0]:.3f},{ci[1]:.3f}]  AUPRC {stats['macro_auprc']:.4f}")
    return result
