"""Monitoring anti-collapse sur la val en fin d'epoch (moyenne des rapports par batch)."""
from __future__ import annotations

import numpy as np
import torch

from ..jepa import JEPA
from ..metrics import collapse_report


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
