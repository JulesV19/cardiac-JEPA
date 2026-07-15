"""Agrège tous les runs/<bb>/<arm>/<regime>/s<seed>/result.json en un seul results.json.

Produit, par (backbone, arm, régime) : moyenne ± écart-type inter-graines du macro-AUROC et
de l'AUPRC, et l'écart PAIRÉ jepa−scratch par graine (moyenne, sd, t de Student). C'est la
source unique des figures et du README — aucun chiffre n'est recopié à la main.

    python -m jepa.aggregate            # écrit runs/results.json + affiche un récap
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np


def _load():
    """(bb, arm, regime, seed) -> dict result. Structure de chemin fixe : runs/bb/arm/regime/sN."""
    out = {}
    for p in glob.glob("runs/*/*/*/s*/result.json"):
        parts = p.split("/")
        bb, arm, regime, sdir = parts[1], parts[2], parts[3], parts[4]
        seed = int(sdir[1:])
        out[(bb, arm, regime, seed)] = json.load(open(p))
    return out


def _stats(vals):
    a = np.array(vals, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a)), "vals": [float(x) for x in a]}


def main() -> None:
    runs = _load()
    if not runs:
        print("Aucun run trouvé sous runs/*/*/*/s*/result.json — lancer la campagne d'abord.")
        return

    # index : (bb, arm, regime) -> {seed: result}
    grouped = defaultdict(dict)
    for (bb, arm, regime, seed), r in runs.items():
        grouped[(bb, arm, regime)][seed] = r

    cells = {}       # "bb|arm|regime" -> stats AUROC/AUPRC
    for (bb, arm, regime), by_seed in grouped.items():
        aur = [by_seed[s]["macro_auroc"] for s in sorted(by_seed)]
        apr = [by_seed[s]["macro_auprc"] for s in sorted(by_seed)]
        cells[f"{bb}|{arm}|{regime}"] = {
            "backbone": bb, "arm": arm, "regime": regime,
            "auroc": _stats(aur), "auprc": _stats(apr),
            "seeds": sorted(by_seed)}

    # écart pairé jepa − scratch par graine, pour chaque (bb, regime)
    gaps = {}
    for (bb, arm, regime), by_seed in grouped.items():
        if arm != "jepa":
            continue
        sc = grouped.get((bb, "scratch", regime))
        if not sc:
            continue
        common = sorted(set(by_seed) & set(sc))
        if not common:
            continue
        d = np.array([by_seed[s]["macro_auroc"] - sc[s]["macro_auroc"] for s in common])
        n = len(d)
        sd = d.std(ddof=1) if n > 1 else 0.0
        t = (d.mean() / (sd / np.sqrt(n))) if (n > 1 and sd > 0) else float("nan")
        gaps[f"{bb}|{regime}"] = {"backbone": bb, "regime": regime, "n": n,
                                  "mean": float(d.mean()), "sd": float(sd), "t": float(t),
                                  "per_seed": [float(x) for x in d]}

    results = {"cells": cells, "paired_gaps": gaps}
    json.dump(results, open("runs/results.json", "w"), indent=2)
    print(f"écrit runs/results.json ({len(cells)} cellules, {len(gaps)} écarts pairés)\n")

    # récap lisible
    order = {"probe": 0, "ft1": 1, "ft5": 2, "ft10": 3, "ft100": 4, "supervised": 5}
    print(f"{'backbone':10} {'arm':9} {'regime':10} {'macro-AUROC':>18} {'AUPRC':>8}  seeds")
    for k in sorted(cells, key=lambda k: (cells[k]['backbone'], order.get(cells[k]['regime'], 9),
                                          cells[k]['arm'])):
        c = cells[k]
        a = c["auroc"]
        print(f"{c['backbone']:10} {c['arm']:9} {c['regime']:10} "
              f"{a['mean']:.4f} ± {a['sd']:.4f} (n={a['n']}) {c['auprc']['mean']:.4f}")
    print(f"\n{'écart pairé JEPA − scratch (macro-AUROC)':40}")
    for k in sorted(gaps, key=lambda k: (gaps[k]['backbone'], order.get(gaps[k]['regime'], 9))):
        g = gaps[k]
        print(f"  {g['backbone']:9} {g['regime']:10} {g['mean']:+.4f} ± {g['sd']:.4f} "
              f"(n={g['n']}, t={g['t']:.2f})")


if __name__ == "__main__":
    main()
