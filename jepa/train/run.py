"""Boucle de pré-entraînement JEPA + monitoring anti-collapse.

Usage :
    python -m jepa.train --config jepa/configs/tiny.yaml
    python -m jepa.train --config jepa/configs/tiny.yaml --epochs 1 --limit 512   # smoke
    python -m jepa.train --out /content/drive/MyDrive/cjepa/run1 --resume auto    # Colab

Sorties dans <out>/ : metrics.csv (train+val+probe), latest.pt (écrit chaque epoch, pour
reprendre après une déconnexion Colab) et best.pt (meilleur epoch selon la sonde linéaire
de sélection, cf. --probe-subsample / --no-probe). Plus de checkpoints périodiques.
Early stopping sur l'AUROC-sonde (--patience, --min-delta).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..data import PTBXLDataset
from ..device import pick_device
from ..jepa import JEPA
from ..losses import total_loss
from ..masking import MaskCollator, MaskConfig
from ..metrics import is_collapsing
from ..models import ModelConfig
from ..progress import tqdm
from ..eval import quick_probe_auroc
from .checkpoint import load_best_score, load_resume, save_best, save_ckpt
from .csvlog import open_metrics_csv
from .monitor import evaluate
from .schedule import build_param_groups, lr_at, momentum_at


def parse_args() -> argparse.Namespace:
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
    ap.add_argument("--stop-epoch", type=int, default=None,
                    help="dernière epoch à exécuter (indice, inclus). Le planning de LR "
                         "reste calculé sur cfg.epochs -> comparaison d'ablation valide.")
    ap.add_argument("--seed", type=int, default=0,
                    help="graine : init des poids, masques, ordre des batches")
    ap.add_argument("--probe-subsample", type=int, default=4000,
                    help="nb d'ECG (folds 1-8) pour la sonde de sélection à chaque epoch")
    ap.add_argument("--no-probe", action="store_true",
                    help="désactive la sonde de sélection (ne sauve alors que latest.pt)")
    ap.add_argument("--patience", type=int, default=15,
                    help="early stopping : arrêt si l'AUROC-sonde ne progresse pas depuis "
                         "N epochs. 0 = désactivé. Sans effet avec --no-probe.")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="progression minimale de l'AUROC-sonde comptée comme une amélioration "
                         "(garde-fou contre le bruit de la sonde)")
    return ap.parse_args()


def load_config(args) -> dict:
    cfg = yaml.safe_load(open(args.config))
    tcfg = cfg["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.workers is not None:
        tcfg["num_workers"] = args.workers
    return cfg


def resolve_out_dir(out_arg) -> Path:
    if out_arg:
        out_dir = Path(out_arg)
        if not out_dir.is_absolute():
            out_dir = Path("runs") / out_arg
    else:
        out_dir = Path("runs") / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_loaders(cfg, tcfg, args):
    model_cfg = ModelConfig(**cfg["model"])
    mask_cfg = MaskConfig(grid_h=model_cfg.grid_h, grid_w=model_cfg.grid_w, **cfg["mask"])

    train_ds = PTBXLDataset("pretrain")
    val_ds = PTBXLDataset("val")
    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit // 4 or 1, len(val_ds))))

    collate = MaskCollator(mask_cfg, seed=args.seed)
    dl_kw = dict(batch_size=tcfg["batch_size"], collate_fn=collate,
                 num_workers=tcfg["num_workers"],
                 persistent_workers=tcfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    # drop_last=False sur val : sinon un val plus petit qu'un batch => 0 batch
    # => monitoring anti-collapse silencieusement absent.
    val_dl = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_kw)
    if len(val_dl) == 0:
        raise RuntimeError("split val vide : le monitoring anti-collapse serait inactif.")
    return model_cfg, train_ds, val_ds, train_dl, val_dl


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args)
    tcfg = cfg["train"]
    device = torch.device(args.device) if args.device else pick_device()
    out_dir = resolve_out_dir(args.out)
    print(f"device={device}  out={out_dir}")

    model_cfg, train_ds, val_ds, train_dl, val_dl = build_loaders(cfg, tcfg, args)

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
    start_epoch, step = load_resume(args.resume, out_dir, model, opt, scaler, device)

    # Reprise du meilleur score connu : après un redémarrage Colab, ne pas écraser best.pt
    # avec un epoch moins bon.
    best_path = out_dir / "best.pt"
    best_auc, best_epoch = load_best_score(best_path)
    epochs_no_improve = 0   # compteur d'early stopping (remis à zéro à la reprise)

    csv_f, writer = open_metrics_csv(out_dir / "metrics.csv", resuming=start_epoch > 0)

    pbar = tqdm(range(start_epoch, tcfg["epochs"]), initial=start_epoch,
                total=tcfg["epochs"], desc=f"{out_dir.parent.name}/{out_dir.name}", unit="ep")
    for epoch in pbar:
        model.train()
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
                                 "", "", "", "", "", "", "", ""])
                csv_f.flush()
            step += 1

        # Monitoring collapse sur val en fin d'epoch.
        rep = evaluate(model, val_dl, device, tcfg["val_batches"])

        # Sonde linéaire de sélection : le SEUL critère fiable du meilleur epoch (la loss/R²
        # JEPA ne mesurent pas la qualité aval). Coûte un forward sur ~probe_subsample ECG.
        probe_auc = None
        if not args.no_probe:
            probe_auc = quick_probe_auroc(model.target_encoder, device,
                                          train_limit=args.probe_subsample,
                                          workers=tcfg["num_workers"])
            model.train()  # extract_features a mis l'encodeur en eval

        if rep:
            writer.writerow(["val", epoch, step, "", "", "", "", "", "",
                             f"{rep['emb_std_ctx']:.4f}", f"{rep['emb_std_tgt']:.4f}",
                             f"{rep['pred_std']:.4f}", f"{rep['eff_rank_ctx']:.2f}",
                             f"{rep['eff_rank_tgt']:.2f}", f"{rep['r2']:.4f}",
                             f"{rep['cos']:.4f}",
                             f"{probe_auc:.4f}" if probe_auc is not None else ""])
            csv_f.flush()
            if is_collapsing(rep):
                tqdm.write(f"  ⚠ COLLAPSE e{epoch} (std ctx={rep['emb_std_ctx']:.3f} "
                           f"rang={rep['eff_rank_ctx']:.1f})")
            pbar.set_postfix(probe=f"{probe_auc:.4f}" if probe_auc is not None else "-",
                             r2=f"{rep['r2']:.2f}", cos=f"{rep['cos']:.2f}",
                             rank=f"{rep['eff_rank_ctx']:.0f}")

        # latest.pt à chaque epoch : une déconnexion Colab ne coûte qu'une epoch.
        save_ckpt(out_dir / "latest.pt", model, opt, scaler, cfg, epoch, step)
        # best.pt : conservé uniquement quand la sonde s'améliore (plus de checkpoints périodiques).
        if probe_auc is not None:
            if probe_auc > best_auc + args.min_delta:
                best_auc, best_epoch = probe_auc, epoch
                save_best(best_path, model, cfg, epoch, step, probe_auc)
                epochs_no_improve = 0
                tqdm.write(f"  ✓ best e{epoch} probe-AUROC {probe_auc:.4f}")
            else:
                epochs_no_improve += 1
                # Early stopping : inutile de continuer si la sonde plafonne.
                if args.patience and epochs_no_improve >= args.patience:
                    tqdm.write(f"  early-stop e{epoch} : {epochs_no_improve} epochs sans progrès "
                               f"(best e{best_epoch} = {best_auc:.4f})")
                    break

        if args.stop_epoch is not None and epoch >= args.stop_epoch:
            break

    pbar.close()
    csv_f.close()
    tag = f"{out_dir.parent.name}/{out_dir.name}"
    tqdm.write(f"  {tag}: best e{best_epoch} probe-AUROC {best_auc:.4f} -> {best_path}"
               if best_epoch >= 0 else f"  {tag}: terminé (pas de sonde)")
