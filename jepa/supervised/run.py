"""Baseline supervisée — classifieur multi-label PTB-XL entraîné from-scratch (xresnet1d101).

Point de référence pour **isoler l'effet backbone** : recette STRICTEMENT identique à la classif
JEPA (`jepa/configs/classify.yaml`), on ne change que la classe d'encodeur. Aucun pré-entraînement,
aucun checkpoint : le modèle part de poids aléatoires, exactement comme le bras `--random-init` du
CNN-JEPA. Le chiffre TEST est donc directement comparable à CNN aléatoire 0,882 / CNN-JEPA 0,888 /
ViT 0,875.

Splits : train folds 1-8, sélection du meilleur epoch sur fold 9 (macro-AUROC), test sur fold 10.
Les ECG sans superclasse sont exclus.

Usage :
    python -m jepa.supervised --config jepa/configs/xresnet.yaml --out runs/xresnet101
    python -m jepa.supervised --out runs/xr_smoke --limit 256 --epochs 2 --device cpu   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from ..classify.labels import labels_of
from ..classify.schedule import build_param_groups, lr_mult
from ..data import SUPERCLASSES, PTBXLDataset
from ..device import pick_device
from ..probe import macro_auroc
from .model import make_xresnet


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> tuple[float, dict, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    model.train()
    logits, ys = np.concatenate(logits), np.concatenate(ys)
    auc, per_class = macro_auroc(ys, logits)
    return auc, per_class, logits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jepa/configs/xresnet.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="sous-ensemble (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="graine : init de la tête, ordre des batches, ET choix du "
                         "sous-échantillon de labels (identique entre les bras)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction des labels d'entraînement (régime peu-de-labels)")
    args = ap.parse_args()
    if not 0 < args.train_frac <= 1:
        ap.error("--train-frac doit être dans ]0, 1]")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_cfg = yaml.safe_load(open(args.config))
    cfg, model_cfg = full_cfg["train"], full_cfg.get("model", {})
    for k, v in [("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("num_workers", args.workers)]:
        if v is not None:
            cfg[k] = v

    device = torch.device(args.device) if args.device else pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Modèle : from-scratch, sans JEPA ---
    model = make_xresnet(**model_cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = model_cfg.get("arch", "xresnet1d101")
    print(f"Baseline supervisée {arch} (from-scratch)\n"
          f"device={device}  params={n_par/1e6:.2f} M  out={out_dir}")

    # --- Données ---
    dsets = {s: PTBXLDataset(s, with_labels=True, drop_unlabeled=True)
             for s in ("pretrain", "val", "test")}
    if args.limit:
        dsets = {s: Subset(d, range(min(args.limit, len(d)))) for s, d in dsets.items()}

    # Régime peu-de-labels : sous-échantillon tiré avec un RNG dédié à la graine (STRICTEMENT
    # identique entre bras — même logique que classify/run.py). Val et test restent complets.
    train_ds = dsets["pretrain"]
    if args.train_frac < 1.0:
        n = max(1, int(round(args.train_frac * len(train_ds))))
        sel = np.sort(np.random.default_rng(args.seed).choice(len(train_ds), n, replace=False))
        train_ds = Subset(train_ds, sel.tolist())
    prev = labels_of(train_ds).sum(axis=0).astype(int)
    print(f"train={len(train_ds)} ({100*args.train_frac:.1f}% des labels, seed={args.seed})"
          f"  val={len(dsets['val'])} test={len(dsets['test'])}")
    print("  positifs par classe : " + "  ".join(f"{c}={n}" for c, n in zip(SUPERCLASSES, prev)))
    if (prev < 2).any():
        print("  ATTENTION : une classe a <2 positifs, son AUROC sera indéfinie.")

    dl_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    drop = len(train_ds) >= 4 * cfg["batch_size"]
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=drop, **dl_kw)
    val_dl = DataLoader(dsets["val"], shuffle=False, **dl_kw)
    test_dl = DataLoader(dsets["test"], shuffle=False, **dl_kw)
    assert len(train_dl) > 0, "batch_size trop grand pour ce sous-échantillon"

    # Recette IDENTIQUE à classify : groupes de params tête/backbone, lr différencié, AdamW.
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg["head_lr"], cfg["encoder_lr"], cfg["weight_decay"]),
        betas=(0.9, 0.999))
    lossf = nn.BCEWithLogitsLoss()
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(len(train_dl), 1)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = int(cfg["warmup_frac"] * total_steps)

    csv_f = open(out_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["epoch", "step", "train_loss", "val_macro_auroc"] + SUPERCLASSES)

    best_auc, best_epoch, step = -1.0, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_mult(step, total_steps, warmup)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = lossf(logits.float(), y)      # loss en fp32, hors autocast
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1

        val_auc, per_class, _ = evaluate(model, val_dl, device)
        writer.writerow([epoch, step, f"{np.mean(losses):.4f}", f"{val_auc:.4f}"]
                        + [f"{per_class.get(c, float('nan')):.4f}" for c in SUPERCLASSES])
        csv_f.flush()

        star = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auroc": val_auc,
                        "arch": arch}, out_dir / "best.pt")
            star = "  <- best"
        print(f"e{epoch:<3} loss={np.mean(losses):.4f}  val_macro_AUROC={val_auc:.4f} "
              f"({time.time()-t0:.0f}s){star}", flush=True)

    # --- Test : uniquement avec le meilleur checkpoint, une seule fois ---
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_auc, test_per_class, _ = evaluate(model, test_dl, device)

    print(f"\nmeilleur epoch : {best_epoch}  (val macro-AUROC {best_auc:.4f})")
    print(f"TEST macro-AUROC = {test_auc:.4f}")
    for c, v in test_per_class.items():
        print(f"  {c:5s} {v:.4f}")

    (out_dir / "result.json").write_text(json.dumps(
        {"arch": arch, "best_epoch": best_epoch, "val_macro_auroc": best_auc,
         "test_macro_auroc": test_auc, "test_per_class": test_per_class}, indent=2))
    csv_f.close()
    print(f"\nrésultats -> {out_dir/'result.json'}")
