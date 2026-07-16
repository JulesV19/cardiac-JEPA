"""Balayage de seuil : évolution des stats de décision en fonction de τ (sur le test).

On NE choisit PAS de seuil (donc aucune fuite val→test) : on trace comment sensibilité,
spécificité, précision et F1 évoluent quand on fait glisser le seuil de décision τ de 0 à 1,
par classe et en macro. Complète l'AUROC/AUPRC (qui, elles, intègrent tous les seuils) par une
lecture « décision clinique » sur le modèle qu'on déploierait.

Charge un `best.pt` déjà entraîné (finetune JEPA/aléatoire, ou supervisé BN), infère sur le
fold 10, et écrit un petit `thresholds.json` (les courbes) — pas de poids, léger à télécharger.
La figure se trace ensuite en local via make_figures.py.

    python -m jepa.eval.thresholds --mode supervised \
        --config jepa/configs/xresnet_supervised.yaml \
        --ckpt runs/xresnet_bn/supervised/ft100/s0/best.pt \
        --label "xresnet BN" --out runs/xresnet_bn/supervised/ft100/s0/thresholds.json

    python -m jepa.eval.thresholds --mode finetune \
        --config jepa/configs/cnn.yaml \
        --ckpt runs/cnn/jepa/ft100/s0/best.pt \
        --label "CNN JEPA" --out runs/cnn/jepa/ft100/s0/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data import SUPERCLASSES, PTBXLDataset
from ..device import pick_device
from ..models import ModelConfig
from ..progress import tqdm
from .build import build_jepa
from .model import ECGClassifier


def load_classifier(mode: str, full_cfg: dict, ckpt: str, device) -> torch.nn.Module:
    """Reconstruit l'architecture (depuis le config) puis charge les poids du best.pt."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    if mode == "supervised":
        from ..models.xresnet_supervised import make_xresnet
        model = make_xresnet(**full_cfg["model"])
    else:                                       # finetune / probe : encodeur JEPA + tête linéaire
        jepa, _ = build_jepa(ModelConfig(**full_cfg["model"]), ckpt=None, random_init=True)
        model = ECGClassifier(jepa.target_encoder, jepa.cfg.embed_dim, len(SUPERCLASSES))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_test(model, device, batch_size: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilités sigmoïde (p̂) et labels sur le fold 10."""
    ds = PTBXLDataset("test", with_labels=True, drop_unlabeled=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    ps, ys = [], []
    for x, y in tqdm(dl, desc="test", leave=False):
        ps.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def sweep(y_true: np.ndarray, y_score: np.ndarray, taus: np.ndarray) -> dict:
    """Pour chaque classe et chaque τ : sensibilité, spécificité, précision, F1 (+ macro)."""
    curves, present = {}, []
    for j, c in enumerate(SUPERCLASSES):
        y, p = y_true[:, j].astype(bool), y_score[:, j]
        if y.sum() == 0 or (~y).sum() == 0:     # classe constante -> métriques indéfinies
            continue
        present.append(c)
        pred = p[None, :] >= taus[:, None]      # (T, N)
        tp = (pred & y).sum(1).astype(float)
        fp = (pred & ~y).sum(1).astype(float)
        fn = (~pred & y).sum(1).astype(float)
        tn = (~pred & ~y).sum(1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            f1 = np.where((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0.0)
        curves[c] = {"sensitivity": sens.tolist(), "specificity": spec.tolist(),
                     "precision": prec.tolist(), "f1": f1.tolist(),
                     "prevalence": float(y.mean())}
    # macro = moyenne sur les classes présentes, τ par τ (nan si tous indéfinis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        macro = {m: np.nanmean([curves[c][m] for c in present], axis=0).tolist()
                 for m in ("sensitivity", "specificity", "precision", "f1")}
    curves["macro"] = macro
    return curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune", "supervised", "probe"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pt entraîné")
    ap.add_argument("--out", required=True, help="chemin du thresholds.json")
    ap.add_argument("--label", default=None, help="nom lisible du modèle (légende)")
    ap.add_argument("--n-tau", type=int, default=101, help="nb de seuils balayés dans [0,1]")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config))
    device = pick_device() if args.device is None else torch.device(args.device)
    model = load_classifier(args.mode, full, args.ckpt, device)
    bs = full.get("eval", {}).get("batch_size", 128)
    y_score, y_true = predict_test(model, device, bs, args.workers)

    taus = np.linspace(0.0, 1.0, args.n_tau)
    curves = sweep(y_true, y_score, taus)
    out = {"label": args.label or Path(args.ckpt).parts[-4], "mode": args.mode,
           "ckpt": args.ckpt, "n_test": int(len(y_true)),
           "thresholds": taus.tolist(), "classes": SUPERCLASSES, "curves": curves}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"écrit {args.out}  ({out['label']}, n_test={out['n_test']}, {args.n_tau} seuils)")


if __name__ == "__main__":
    main()
