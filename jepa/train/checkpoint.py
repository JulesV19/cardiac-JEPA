"""Sauvegarde / reprise des checkpoints de pré-entraînement.

- latest.pt : modèle + optimiseur + scaler + position (reprise après déconnexion Colab).
- best.pt   : léger (pas d'optimiseur), écrit quand la sonde de sélection s'améliore.
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_ckpt(path: Path, model, opt, scaler, cfg, epoch: int, step: int) -> None:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "step": step,
                "cfg": cfg}, path)


def save_best(path: Path, model, cfg, epoch: int, step: int, probe_auroc: float) -> None:
    # Léger : pas d'optimiseur (best.pt sert à l'évaluation aval, pas à reprendre).
    torch.save({"model": model.state_dict(), "epoch": epoch, "step": step,
                "cfg": cfg, "probe_auroc": probe_auroc}, path)


def load_resume(resume_arg, out_dir: Path, model, opt, scaler, device) -> tuple[int, int]:
    """Reprend modèle+optimiseur+scaler et renvoie (start_epoch, step). (0, 0) si pas de reprise."""
    resume_path = out_dir / "latest.pt" if resume_arg == "auto" else (
        Path(resume_arg) if resume_arg else None)
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"reprise depuis {resume_path} : epoch {start_epoch}, step {step}")
        return start_epoch, step
    if resume_arg and resume_arg != "auto":
        raise FileNotFoundError(f"checkpoint introuvable : {resume_path}")
    return 0, 0


def load_best_score(best_path: Path) -> tuple[float, int]:
    """Reprend le meilleur score connu pour ne pas écraser best.pt avec un epoch moins bon."""
    if best_path.exists():
        prev = torch.load(best_path, map_location="cpu", weights_only=False)
        best_auc, best_epoch = prev.get("probe_auroc", -1.0), prev.get("epoch", -1)
        print(f"best.pt existant : epoch {best_epoch}, probe-AUROC {best_auc:.4f}")
        return best_auc, best_epoch
    return -1.0, -1
