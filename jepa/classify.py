"""Phase 2 — classifieur multi-label PTB-XL par fine-tuning de l'encodeur JEPA.

Encodeur cible EMA (initialisé depuis le pré-entraînement) + moyenne des 480 tokens
+ Linear(192 -> 5). **Fine-tuning complet**, avec un lr 10x plus faible sur l'encodeur
que sur la tête : le pré-entraînement sert d'initialisation, pas de contrainte.

Splits : train folds 1-8, sélection du meilleur epoch sur fold 9 (macro-AUROC),
résultat final rapporté sur fold 10. Les 407 ECG sans superclasse sont exclus.
Le meilleur checkpoint (best.pt) est écrit dès qu'il s'améliore.

Contrôle indispensable : `--random-init` fine-tune un encodeur NON pré-entraîné.
Si le JEPA ne le bat pas, le pré-entraînement n'a servi à rien — et l'AUROC seule
ne le dirait pas.

Usage :
    python -m jepa.classify --ckpt runs/tiny_v1/ckpt_e99.pt --out runs/clf_jepa
    python -m jepa.classify --random-init --out runs/clf_random     # contrôle
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from .data import SUPERCLASSES, PTBXLDataset
from .jepa import JEPA
from .models import ModelConfig
from .probe import macro_auroc
from .train import pick_device


class ECGClassifier(nn.Module):
    """Encodeur JEPA -> moyenne des tokens -> tête linéaire multi-label."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, None)          # (B, 480, D), aucun masquage
        return self.head(z.mean(dim=1))    # (B, n_classes)


def labels_of(ds) -> np.ndarray:
    """Matrice multi-hot (N, 5) d'un PTBXLDataset ou d'un Subset imbriqué.

    Lit `labels` directement : ne charge aucun signal.
    """
    if isinstance(ds, Subset):
        return labels_of(ds.dataset)[np.asarray(ds.indices)]
    return ds.labels[ds.positions]


def build_param_groups(model: ECGClassifier, head_lr: float, encoder_lr: float,
                       weight_decay: float):
    """4 groupes : {encodeur, tête} x {avec, sans weight decay}.

    Pas de weight decay sur les params 1D (norms, biais, pos embeds).
    `base_lr` est mémorisé par groupe : le scheduler applique un multiplicateur commun.
    """
    groups = {
        ("enc", True):  {"params": [], "weight_decay": weight_decay, "base_lr": encoder_lr},
        ("enc", False): {"params": [], "weight_decay": 0.0,          "base_lr": encoder_lr},
        ("head", True): {"params": [], "weight_decay": weight_decay, "base_lr": head_lr},
        ("head", False):{"params": [], "weight_decay": 0.0,          "base_lr": head_lr},
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if n.startswith("head") else "enc"
        decay = not (p.ndim <= 1 or "pos" in n)
        groups[(part, decay)]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def lr_mult(step: int, total: int, warmup: int) -> float:
    """Multiplicateur de lr : warmup linéaire puis cosine."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model: ECGClassifier, loader, device) -> tuple[float, dict, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    model.train()
    logits, ys = np.concatenate(logits), np.concatenate(ys)
    auc, per_class = macro_auroc(ys, logits)
    return auc, per_class, logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint de pré-entraînement JEPA")
    ap.add_argument("--random-init", action="store_true",
                    help="contrôle : encodeur NON pré-entraîné, fine-tuné pareil")
    ap.add_argument("--encoder", choices=["target", "online"], default="target")
    ap.add_argument("--config", default="jepa/configs/classify.yaml")
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
    if not args.ckpt and not args.random_init:
        ap.error("donne --ckpt, ou --random-init pour le contrôle")
    if not 0 < args.train_frac <= 1:
        ap.error("--train-frac doit être dans ]0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(open(args.config))["train"]
    for k, v in [("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("num_workers", args.workers)]:
        if v is not None:
            cfg[k] = v

    device = torch.device(args.device) if args.device else pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Encodeur : pré-entraîné ou aléatoire (même architecture) ---
    if args.random_init:
        jepa = JEPA(ModelConfig())
        tag = "random-init (contrôle)"
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        jepa = JEPA(ModelConfig(**ck["cfg"]["model"]))
        jepa.load_state_dict(ck["model"])
        tag = f"{args.ckpt} (epoch {ck['epoch']}, encodeur {args.encoder})"
    encoder = jepa.target_encoder if args.encoder == "target" else jepa.encoder
    # target_encoder a requires_grad=False (stop-grad du pré-entraînement) : on le dégèle,
    # sinon le « fine-tuning complet » n'entraînerait silencieusement que la tête.
    for p in encoder.parameters():
        p.requires_grad_(True)

    model = ECGClassifier(encoder, jepa.cfg.embed_dim, len(SUPERCLASSES)).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert n_enc > 0, "l'encodeur est gelé : le fine-tuning ne ferait rien"
    print(f"Fine-tuning depuis {tag}\ndevice={device}  params={n_par/1e6:.2f} M  out={out_dir}")

    # --- Données ---
    dsets = {s: PTBXLDataset(s, with_labels=True, drop_unlabeled=True)
             for s in ("pretrain", "val", "test")}
    if args.limit:
        dsets = {s: Subset(d, range(min(args.limit, len(d)))) for s, d in dsets.items()}

    # Régime peu-de-labels : sous-échantillon tiré avec un RNG dédié à la graine, donc
    # STRICTEMENT identique entre le bras JEPA et le bras aléatoire (sinon la comparaison
    # ne veut rien dire). Val et test restent complets.
    train_ds = dsets["pretrain"]
    if args.train_frac < 1.0:
        n = max(1, int(round(args.train_frac * len(train_ds))))
        sel = np.sort(np.random.default_rng(args.seed).choice(len(train_ds), n, replace=False))
        train_ds = Subset(train_ds, sel.tolist())
    y_tr = labels_of(train_ds)
    prev = y_tr.sum(axis=0).astype(int)
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

    opt = torch.optim.AdamW(
        build_param_groups(model, cfg["head_lr"], cfg["encoder_lr"], cfg["weight_decay"]),
        betas=(0.9, 0.999))
    lossf = nn.BCEWithLogitsLoss()          # pas de pondération : macro-AUROC = métrique de rang
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
                        "model_cfg": vars(jepa.cfg), "tag": tag}, out_dir / "best.pt")
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
        {"tag": tag, "best_epoch": best_epoch, "val_macro_auroc": best_auc,
         "test_macro_auroc": test_auc, "test_per_class": test_per_class}, indent=2))
    csv_f.close()
    print(f"\nrésultats -> {out_dir/'result.json'}")


if __name__ == "__main__":
    main()
