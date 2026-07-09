"""Boucle de pré-entraînement JEPA + monitoring anti-collapse.

Usage :
    python -m jepa.train --config jepa/configs/tiny.yaml
    python -m jepa.train --config jepa/configs/tiny.yaml --epochs 1 --limit 512   # smoke
    python -m jepa.train --out /content/drive/MyDrive/cjepa/run1 --resume auto    # Colab

Sorties dans <out>/ : metrics.csv (train+val), latest.pt (écrit chaque epoch, pour
reprendre après une déconnexion Colab) et ckpt_e<N>.pt périodiques.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .data import PTBXLDataset
from .jepa import JEPA
from .losses import total_loss
from .masking import MaskCollator, MaskConfig
from .metrics import collapse_report, is_collapsing
from .models import ModelConfig


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Pas de weight decay sur les params 1D (norms, biais, pos embeds, mask token)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "pos" in n or "mask_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def momentum_at(step: int, total: int, m0: float, m1: float) -> float:
    return m0 + (m1 - m0) * step / max(total - 1, 1)


@torch.no_grad()
def evaluate(model: JEPA, loader, device, n_batches: int) -> dict:
    model.eval()
    reps = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        sig = batch["signals"].to(device)
        cidx = batch["context_idx"].to(device)
        tidx = batch["target_idx"].to(device)
        pred, z_tgt, z_ctx = model(sig, cidx, tidx)
        reps.append(collapse_report(z_ctx, z_tgt, pred))
    model.train()
    if not reps:
        return {}
    return {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="jepa/configs/tiny.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="limite le nb d'ECG (smoke test)")
    ap.add_argument("--out", default=None,
                    help="dossier de sortie (chemin absolu accepté, ex. sur Drive)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="chemin d'un checkpoint, ou 'auto' pour <out>/latest.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = cfg["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.workers is not None:
        tcfg["num_workers"] = args.workers

    device = torch.device(args.device) if args.device else pick_device()
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = Path("runs") / args.out
    else:
        out_dir = Path("runs") / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    model_cfg = ModelConfig(**cfg["model"])
    mask_cfg = MaskConfig(grid_h=model_cfg.grid_h, grid_w=model_cfg.grid_w, **cfg["mask"])

    train_ds = PTBXLDataset("pretrain")
    val_ds = PTBXLDataset("val")
    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit // 4 or 1, len(val_ds))))

    collate = MaskCollator(mask_cfg)
    dl_kw = dict(batch_size=tcfg["batch_size"], collate_fn=collate,
                 num_workers=tcfg["num_workers"], drop_last=True,
                 persistent_workers=tcfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    model = JEPA(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params entraînables : {n_params/1e6:.2f} M | train={len(train_ds)} val={len(val_ds)}")

    base_lr = tcfg["base_lr"] * tcfg["batch_size"] / 256
    opt = torch.optim.AdamW(build_param_groups(model, tcfg["weight_decay"]),
                            lr=base_lr, betas=(0.9, 0.95))
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = tcfg["epochs"] * steps_per_epoch
    warmup = int(tcfg["warmup_frac"] * total_steps)

    use_amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Reprise après déconnexion (Colab) : modèle + optimiseur + scaler + position.
    start_epoch, step = 0, 0
    resume_path = out_dir / "latest.pt" if args.resume == "auto" else (
        Path(args.resume) if args.resume else None)
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"reprise depuis {resume_path} : epoch {start_epoch}, step {step}")
    elif args.resume and args.resume != "auto":
        raise FileNotFoundError(f"checkpoint introuvable : {resume_path}")

    def save_ckpt(path: Path, epoch: int) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step,
                    "cfg": cfg}, path)

    csv_path = out_dir / "metrics.csv"
    new_csv = not (csv_path.exists() and start_epoch > 0)
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["phase", "epoch", "step", "lr", "momentum", "total", "jepa",
                         "var", "cov", "emb_std_ctx", "emb_std_tgt", "pred_std",
                         "eff_rank_ctx", "eff_rank_tgt"])

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0 = time.time()
        for batch in train_dl:
            lr = lr_at(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            m = momentum_at(step, total_steps, tcfg["ema_start"], tcfg["ema_end"])

            sig = batch["signals"].to(device)
            cidx = batch["context_idx"].to(device)
            tidx = batch["target_idx"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred, z_tgt, z_ctx = model(sig, cidx, tidx)
            # Loss hors autocast : VICReg déborde en fp16 (cf. jepa/losses.py).
            loss, parts = total_loss(pred, z_tgt, z_ctx,
                                     tcfg["lambda_var"], tcfg["lambda_cov"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.update_target(m)

            if step % tcfg["log_every"] == 0:
                writer.writerow(["train", epoch, step, f"{lr:.2e}", f"{m:.5f}",
                                 f"{parts['total']:.4f}", f"{parts['jepa']:.4f}",
                                 f"{parts['var']:.4f}", f"{parts['cov']:.4f}",
                                 "", "", "", "", ""])
                csv_f.flush()
                print(f"e{epoch} s{step} lr{lr:.1e} m{m:.4f} "
                      f"L{parts['total']:.3f} jepa{parts['jepa']:.3f} "
                      f"var{parts['var']:.3f} cov{parts['cov']:.4f}", flush=True)
            step += 1

        # Monitoring collapse sur val en fin d'epoch.
        rep = evaluate(model, val_dl, device, tcfg["val_batches"])
        if rep:
            writer.writerow(["val", epoch, step, "", "", "", "", "", "",
                             f"{rep['emb_std_ctx']:.4f}", f"{rep['emb_std_tgt']:.4f}",
                             f"{rep['pred_std']:.4f}", f"{rep['eff_rank_ctx']:.2f}",
                             f"{rep['eff_rank_tgt']:.2f}"])
            csv_f.flush()
            flag = "  ⚠ COLLAPSE" if is_collapsing(rep) else ""
            print(f"[val e{epoch}] emb_std_ctx={rep['emb_std_ctx']:.4f} "
                  f"eff_rank_ctx={rep['eff_rank_ctx']:.2f} "
                  f"({time.time()-t0:.1f}s){flag}", flush=True)

        # latest.pt à chaque epoch : une déconnexion Colab ne coûte qu'une epoch.
        save_ckpt(out_dir / "latest.pt", epoch)
        if (epoch + 1) % tcfg["ckpt_every"] == 0 or epoch + 1 == tcfg["epochs"]:
            save_ckpt(out_dir / f"ckpt_e{epoch}.pt", epoch)

    csv_f.close()
    print(f"Terminé. Métriques : {csv_path}")


if __name__ == "__main__":
    main()
