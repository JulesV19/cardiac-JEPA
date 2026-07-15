"""Fine-tuning complet avec EARLY STOPPING sur la val macro-AUROC.

Encodeur (pré-entraîné JEPA ou aléatoire iso-archi) + moyenne des tokens + tête linéaire.
lr 10× plus faible sur l'encodeur que sur la tête. On arrête dès que la val macro-AUROC ne
progresse plus de `min_delta` pendant `patience` epochs (le pic est typiquement à e6-12) ;
`max_epochs` reste un plafond dur. Test = fold 10, une seule fois, avec le meilleur checkpoint.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..data import SUPERCLASSES, PTBXLDataset
from ..models import ModelConfig
from ..progress import tqdm
from .build import build_jepa
from .metrics import macro_auroc, summarize
from .model import ECGClassifier
from .schedule import build_param_groups, lr_mult


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    model.train()
    return np.concatenate(logits), np.concatenate(ys)


def run_finetune(model_cfg: ModelConfig, ckpt, random_init, encoder, cfg: dict,
                 out_dir: Path, seed: int, train_frac: float, device, limit=None) -> dict:
    jepa, tag = build_jepa(model_cfg, ckpt, random_init, encoder)
    enc = jepa.target_encoder if encoder == "target" else jepa.encoder
    for p in enc.parameters():
        p.requires_grad_(True)                       # fine-tuning complet
    model = ECGClassifier(enc, jepa.cfg.embed_dim, len(SUPERCLASSES))
    return fit_classifier(model, tag, "finetune", cfg, out_dir, seed, train_frac, device, limit)


def fit_classifier(model: nn.Module, tag: str, mode: str, cfg: dict, out_dir: Path,
                   seed: int, train_frac: float, device, limit=None) -> dict:
    """Boucle commune (finetune / supervised) : early-stopping val-AUROC, test 1× + stats."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dsets = {s: PTBXLDataset(s, with_labels=True, drop_unlabeled=True)
             for s in ("pretrain", "val", "test")}
    if limit:
        dsets = {s: Subset(d, range(min(limit, len(d)))) for s, d in dsets.items()}
    train_ds = dsets["pretrain"]
    if train_frac < 1.0:                             # sous-échantillon identique entre bras (par graine)
        n = max(1, int(round(train_frac * len(train_ds))))
        sel = np.sort(np.random.default_rng(seed).choice(len(train_ds), n, replace=False))
        train_ds = Subset(train_ds, sel.tolist())
    dl_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    drop = len(train_ds) >= 4 * cfg["batch_size"]
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=drop, **dl_kw)
    val_dl = DataLoader(dsets["val"], shuffle=False, **dl_kw)
    test_dl = DataLoader(dsets["test"], shuffle=False, **dl_kw)
    assert len(train_dl) > 0, "batch_size trop grand pour ce sous-échantillon"

    opt = torch.optim.AdamW(
        build_param_groups(model, cfg["head_lr"], cfg["encoder_lr"], cfg["weight_decay"]),
        betas=(0.9, 0.999))
    lossf = nn.BCEWithLogitsLoss()
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = cfg["max_epochs"] * max(len(train_dl), 1)
    warmup = int(cfg["warmup_frac"] * total_steps)

    csv_f = open(out_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["epoch", "step", "train_loss", "val_macro_auroc"] + SUPERCLASSES)

    label = f"{mode[:4]} f{int(round(train_frac*100))} s{seed} ({n_par/1e6:.1f}M)"
    best_auc, best_epoch, step, since = -1.0, -1, 0, 0
    pbar = tqdm(range(cfg["max_epochs"]), desc=label, leave=False, unit="ep")
    for epoch in pbar:
        model.train()
        losses = []
        for x, y in train_dl:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_mult(step, total_steps, warmup)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = lossf(logits.float(), y)          # loss fp32, hors autocast
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1

        vlog, vy = _evaluate(model, val_dl, device)
        val_auc, per_class = macro_auroc(vy, vlog)
        writer.writerow([epoch, step, f"{np.mean(losses):.4f}", f"{val_auc:.4f}"]
                        + [f"{per_class.get(c, float('nan')):.4f}" for c in SUPERCLASSES])
        csv_f.flush()

        improved = val_auc > best_auc + cfg.get("min_delta", 0.0)
        if val_auc > best_auc:                       # on garde le meilleur strict (best.pt)
            best_auc, best_epoch = val_auc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auroc": val_auc,
                        "tag": tag}, out_dir / "best.pt")
        since = 0 if improved else since + 1
        pbar.set_postfix(loss=f"{np.mean(losses):.3f}", val=f"{val_auc:.4f}",
                         best=f"{best_auc:.4f}@{best_epoch}")
        if since >= cfg["patience"]:
            break
    pbar.close()

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    tlog, ty = _evaluate(model, test_dl, device)
    stats = summarize(ty, tlog, n_boot=cfg.get("n_boot", 2000), seed=seed)
    result = {"tag": tag, "mode": mode, "train_frac": train_frac, "seed": seed,
              "best_epoch": best_epoch, "val_macro_auroc": best_auc, **stats}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    csv_f.close()
    ci = stats["auroc_ci95"]
    tqdm.write(f"  {label:28} AUROC {stats['macro_auroc']:.4f} "
               f"[{ci[0]:.3f},{ci[1]:.3f}]  AUPRC {stats['macro_auprc']:.4f}  (best e{best_epoch})")
    return result
