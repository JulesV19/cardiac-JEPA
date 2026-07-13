"""Fichier metrics.csv : schéma commun train+val+probe et ouverture (avec reprise)."""
from __future__ import annotations

import csv
from pathlib import Path

HEADER = ["phase", "epoch", "step", "lr", "momentum", "total", "jepa",
          "var", "cov", "emb_std_ctx", "emb_std_tgt", "pred_std",
          "eff_rank_ctx", "eff_rank_tgt", "r2", "cos", "probe_auroc"]


def open_metrics_csv(csv_path: Path, resuming: bool):
    """Ouvre metrics.csv en écriture ou append. Renvoie (fichier, writer).

    Si on reprend mais que le schéma a changé (nouvelles colonnes), on archive l'ancien
    fichier plutôt que de mélanger les schémas.
    """
    append = csv_path.exists() and resuming
    if append:
        with open(csv_path) as f:
            old = f.readline().strip().split(",")
        if old != HEADER:
            backup = csv_path.with_suffix(".prev.csv")
            csv_path.rename(backup)
            print(f"schéma CSV modifié -> ancien fichier archivé dans {backup.name}")
            append = False
    csv_f = open(csv_path, "a" if append else "w", newline="")
    writer = csv.writer(csv_f)
    if not append:
        writer.writerow(HEADER)
    return csv_f, writer
