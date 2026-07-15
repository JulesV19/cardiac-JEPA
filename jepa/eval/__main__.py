"""Entrée unifiée d'évaluation : `python -m jepa.eval --mode {probe,finetune}`.

Exemples :
    # sonde gelée, bras JEPA
    python -m jepa.eval --mode probe    --config jepa/configs/xresnet.yaml --ckpt runs/xresnet/pretrain/best.pt --out runs/xresnet/jepa/probe/s0
    # fine-tuning, bras aléatoire iso-archi, 5% des labels
    python -m jepa.eval --mode finetune --config jepa/configs/xresnet.yaml --random-init --train-frac 0.05 --seed 0 --out runs/xresnet/scratch/ft005/s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..device import pick_device
from ..models import ModelConfig
from .finetune import run_finetune
from .probe import run_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "finetune", "supervised"], required=True)
    ap.add_argument("--config", required=True, help="config du backbone (section model: + eval:)")
    ap.add_argument("--ckpt", default=None, help="checkpoint JEPA pré-entraîné (bras JEPA)")
    ap.add_argument("--random-init", action="store_true", help="bras aléatoire iso-archi (contrôle)")
    ap.add_argument("--encoder", choices=["target", "online"], default="target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0, help="finetune : fraction des labels")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="sous-ensemble (smoke test)")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    if args.mode != "supervised" and not args.random_init and not args.ckpt:
        ap.error("donne --ckpt (bras JEPA), ou --random-init (bras aléatoire iso-archi)")
    if not 0 < args.train_frac <= 1:
        ap.error("--train-frac doit être dans ]0, 1]")

    full = yaml.safe_load(open(args.config))
    cfg = full["eval"]
    if args.workers is not None:
        cfg["num_workers"] = args.workers
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    device = pick_device() if args.device is None else __import__("torch").device(args.device)
    out = Path(args.out)

    if args.mode == "supervised":
        from .supervised import run_supervised
        run_supervised(full["model"], cfg, out, args.seed, args.train_frac, device, args.limit)
        return

    model_cfg = ModelConfig(**full["model"])
    if args.mode == "probe":
        run_probe(model_cfg, args.ckpt, args.random_init, args.encoder, out,
                  args.seed, device, workers=cfg["num_workers"], limit=args.limit)
    else:
        run_finetune(model_cfg, args.ckpt, args.random_init, args.encoder, cfg, out,
                     args.seed, args.train_frac, device, limit=args.limit)


if __name__ == "__main__":
    main()
