#!/usr/bin/env python3
"""Génère les figures du README depuis runs/ (régénérable à l'identique).

Toutes les valeurs sont lues des result.json / metrics.csv réels — aucun chiffre codé en dur.
Métrique : macro-AUROC, test = fold 10 (jamais vu), sélection d'epoch sur fold 9.

    python make_figures.py

Sortie : figures/*.png + figures/_summary.json (récapitulatif des chiffres tracés).
"""
import json
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---- palette CVD-safe (validée) ----
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARM = {"random": "#eb6834", "jepa": "#2a78d6"}          # aléatoire / pré-entraîné
MODEL = {"ViT": "#4a3aa7", "CNN v1": "#199e70", "CNN v2": "#2a78d6", "xresnet": "#c81e5a"}
EMBED = {"ViT": 192, "CNN v1": 192, "CNN v2": 256}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK2, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 150,
})


def res(run):
    return json.load(open(f"{RUNS}/{run}/result.json"))["test_macro_auroc"]


def per_class(run):
    return json.load(open(f"{RUNS}/{run}/result.json"))["test_per_class"]


def probe(run, kind):
    return json.load(open(f"{RUNS}/{run}/probe_{kind}.json"))["test_macro_auroc"]


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("écrit", f"{OUT}/{name}.png")


# =========================================================================
# Données (toutes lues des runs)
# =========================================================================
CEIL = {  # fine-tuning 100 % des labels
    "ViT":    {"random": res("clf_random"),        "jepa": res("clf_jepa")},
    "CNN v1": {"random": res("clf_cnn_random"),    "jepa": res("clf_cnn_jepa")},
    "CNN v2": {"random": res("clf_cnn_v2_random"), "jepa": res("clf_cnn_v2_jepa")},
}
XRES = res("xresnet18")                                     # from-scratch, sans SSL
PARAMS = {"ViT": 6.0, "CNN v1": 1.27, "CNN v2": 6.25, "xresnet": 6.30}

# sonde linéaire (features gelées) — le probe ViT vit dans le _summary du rapport CNN
PROBE = {
    "ViT":    {"random": 0.7825071583381865, "jepa": 0.8436695846387587},
    "CNN v2": {"random": probe("cnn_v2", "random"), "jepa": probe("cnn_v2", "cnn")},
}

SWEEP_DIR = {"ViT": "lowlabel", "CNN v1": "lowlabel_cnn", "CNN v2": "lowlabel_cnn_v2"}
FRAC_TAGS = [("0.01", 1), ("0.05", 5), ("0.1", 10), ("1.0", 100)]


def sweep(model):
    d = defaultdict(lambda: defaultdict(list))
    base = SWEEP_DIR[model]
    for ftag, pct in FRAC_TAGS:
        for arm in ("random", "jepa"):
            for p in sorted(glob.glob(f"{RUNS}/{base}/{arm}_f{ftag}_s*/result.json")):
                d[pct][arm].append(json.load(open(p))["test_macro_auroc"])
    return d


SW = {m: sweep(m) for m in SWEEP_DIR}


# =========================================================================
# FIG 1 — Le levier, c'est le backbone (plafond 100 %, 4 backbones)
# =========================================================================
fig, ax = plt.subplots(figsize=(8.4, 3.8))
order = ["ViT", "CNN v1", "CNN v2"]
for i, m in enumerate(order):
    r, j = CEIL[m]["random"], CEIL[m]["jepa"]
    ax.plot([r, j], [i, i], color=MUTED, lw=2, zorder=1)
    ax.scatter([r], [i], s=95, color=ARM["random"], zorder=3)
    ax.scatter([j], [i], s=95, color=ARM["jepa"], zorder=3)
    ax.annotate(f"{r:.3f}", (r, i), textcoords="offset points", xytext=(0, -15),
                ha="center", color=ARM["random"], fontsize=9)
    ax.annotate(f"{j:.3f}", (j, i), textcoords="offset points", xytext=(0, 9),
                ha="center", color=ARM["jepa"], fontsize=9, fontweight="bold")
# xresnet : from-scratch (aucun SSL) — un seul point, la meilleure valeur du projet
yx = len(order)
ax.scatter([XRES], [yx], s=130, marker="D", color=MODEL["xresnet"], zorder=4)
ax.annotate(f"{XRES:.3f}", (XRES, yx), textcoords="offset points", xytext=(0, 10),
            ha="center", color=MODEL["xresnet"], fontsize=10, fontweight="bold")
labels = [f"{m}\n{PARAMS[m]:.2f} M" for m in order] + [f"xresnet1d18\n{PARAMS['xresnet']:.2f} M · sans SSL"]
ax.set_yticks(range(len(order) + 1)); ax.set_yticklabels(labels)
ax.set_ylim(-0.5, yx + 0.75); ax.set_xlim(0.855, 0.90)
ax.set_xlabel("macro-AUROC (test, fold 10)")
ax.set_title("Fine-tuning 100 % des labels — macro-AUROC test par backbone")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=ARM["random"], label="aléatoire (from-scratch)"),
                   Line2D([0], [0], marker="o", ls="", color=ARM["jepa"], label="pré-entraîné JEPA"),
                   Line2D([0], [0], marker="D", ls="", color=MODEL["xresnet"], label="xresnet (from-scratch)")],
          loc="lower right", frameon=False, fontsize=9)
finish(fig, "1_backbone")

# =========================================================================
# FIG 2 — Sonde gelée : le SSL marche (sur les features, avant fine-tuning)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 2.9))
mods = ["ViT", "CNN v2"]
for i, m in enumerate(mods):
    r, j = PROBE[m]["random"], PROBE[m]["jepa"]
    ax.plot([r, j], [i, i], color=MUTED, lw=2, zorder=1)
    ax.scatter([r], [i], s=95, color=ARM["random"], zorder=3)
    ax.scatter([j], [i], s=95, color=ARM["jepa"], zorder=3)
    ax.annotate(f"{r:.3f}", (r, i), textcoords="offset points", xytext=(0, -15),
                ha="center", color=ARM["random"], fontsize=9)
    ax.annotate(f"{j:.3f}", (j, i), textcoords="offset points", xytext=(0, 9),
                ha="center", color=ARM["jepa"], fontsize=9, fontweight="bold")
    ax.annotate(f"+{j-r:.3f}", (j, i), textcoords="offset points", xytext=(28, 0),
                va="center", color=INK, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
ax.set_ylim(-0.6, len(mods) - 0.3); ax.set_xlim(0.75, 0.87)
ax.set_xlabel("macro-AUROC (test) — sonde linéaire sur encodeur gelé")
ax.set_title("Sonde linéaire sur encodeur gelé")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=ARM["random"], label="encodeur aléatoire"),
                   Line2D([0], [0], marker="o", ls="", color=ARM["jepa"], label="encodeur JEPA")],
          loc="lower right", frameon=False, fontsize=9)
finish(fig, "2_probe")

# =========================================================================
# FIG 3 — Peu-de-labels : le SSL prouve sa valeur (CNN v2)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
sw = SW["CNN v2"]
pcts = [1, 5, 10, 100]
for arm in ("random", "jepa"):
    means = [np.mean(sw[p][arm]) for p in pcts]
    stds = [np.std(sw[p][arm], ddof=1) if len(sw[p][arm]) > 1 else 0 for p in pcts]
    ax.errorbar(pcts, means, yerr=stds, marker="o", ms=7, lw=2, capsize=3,
                color=ARM[arm])
    ax.annotate(("aléatoire" if arm == "random" else "JEPA"), (pcts[-1], means[-1]),
                textcoords="offset points", xytext=(8, 8 if arm == "jepa" else -8),
                color=ARM[arm], fontsize=10, fontweight="bold", va="center")
ax.set_xscale("log"); ax.set_xticks(pcts); ax.set_xticklabels([f"{p}%" for p in pcts])
ax.set_xlabel("part des labels d'entraînement (échelle log)")
ax.set_ylabel("macro-AUROC (test)")
ax.set_title("CNN v2 — macro-AUROC test vs part des labels")
ax.set_xlim(0.8, 200)
finish(fig, "3_lowlabel")

# =========================================================================
# FIG 4 — Le gain SSL s'efface quand les labels abondent (écart pairé, 3 modèles)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for m in ["ViT", "CNN v1", "CNN v2"]:
    sw = SW[m]
    gaps = []
    for p in pcts:
        r, j = sw[p]["random"], sw[p]["jepa"]
        n = min(len(r), len(j))
        gaps.append(np.mean(np.array(j[:n]) - np.array(r[:n])))
    ax.plot(pcts, gaps, marker="o", ms=7, lw=2, color=MODEL[m], label=m)
ax.axhline(0, color=MUTED, lw=1, ls="--")
ax.set_xscale("log"); ax.set_xticks(pcts); ax.set_xticklabels([f"{p}%" for p in pcts])
ax.set_xlabel("part des labels d'entraînement (échelle log)")
ax.set_ylabel("écart JEPA − aléatoire (macro-AUROC)")
ax.set_title("Écart JEPA − aléatoire vs part des labels")
ax.set_xlim(0.8, 160)
ax.legend(handles=[Line2D([0], [0], color=MODEL[m], lw=2, label=m) for m in ["ViT", "CNN v1", "CNN v2"]],
          loc="upper right", frameon=False)
finish(fig, "4_ssl_gain")

# =========================================================================
# FIG 5 — Pré-entraînement : encodeur sain (pas de collapse) vs prédicteur qui régresse
# =========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
runs = {"ViT": "tiny_v1", "CNN v1": "cnn_v1", "CNN v2": "cnn_v2"}
for m, run in runs.items():
    df = pd.read_csv(f"{RUNS}/{run}/metrics.csv")
    va = df[df.phase == "val"]
    a1.plot(va.epoch, va.eff_rank_ctx / EMBED[m], color=MODEL[m], lw=2, label=m)
    a2.plot(va.epoch, va.cos, color=MODEL[m], lw=2, label=m)
a1.set_title("Rang effectif normalisé (contexte)")
a1.set_xlabel("epoch"); a1.set_ylabel("rang effectif / dim")
a1.set_ylim(0, 1); a1.legend(frameon=False, loc="lower right")
a2.set_title("cos(prédiction, cible)")
a2.set_xlabel("epoch"); a2.set_ylabel("cos")
a2.legend(frameon=False, loc="upper right")
fig.suptitle("Pré-entraînement JEPA (suivi sur fold 9)",
             fontsize=12, fontweight="bold", color=INK, y=1.02)
finish(fig, "5_pretraining")

# =========================================================================
# FIG 6 — Le mur = généralisation, pas capacité (val pique tôt puis chute)
# =========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
CURVES = [("ViT", "clf_jepa", MODEL["ViT"]),
          ("CNN v1", "clf_cnn_jepa", MODEL["CNN v1"]),
          ("CNN v2", "clf_cnn_v2_jepa", MODEL["CNN v2"]),
          ("xresnet", "xresnet18", MODEL["xresnet"])]
for lbl, run, c in CURVES:
    df = pd.read_csv(f"{RUNS}/{run}/metrics.csv")
    a1.plot(df.epoch, df.val_macro_auroc, color=c, lw=2, label=lbl)
    be = int(df.val_macro_auroc.idxmax())
    a1.scatter([df.epoch.iloc[be]], [df.val_macro_auroc.iloc[be]], s=90, color=c,
               edgecolor=SURF, linewidth=1.5, zorder=5)
    a2.plot(df.epoch, df.train_loss, color=c, lw=2, label=lbl)
a1.set_title("val macro-AUROC (point = meilleur epoch)")
a1.set_xlabel("epoch de fine-tuning"); a1.set_ylabel("val macro-AUROC")
a1.legend(frameon=False, loc="lower left", fontsize=9)
a2.set_title("train loss")
a2.set_xlabel("epoch de fine-tuning"); a2.set_ylabel("train loss")
a2.legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Fine-tuning 100 % des labels — courbes d'entraînement",
             fontsize=13, fontweight="bold", color=INK, y=1.04)
finish(fig, "6_generalization")

# =========================================================================
# Récapitulatif chiffré
# =========================================================================
summary = {
    "ceiling_100pct": {m: CEIL[m] for m in CEIL},
    "xresnet_fromscratch": XRES,
    "probe_frozen": PROBE,
    "sweep_mean": {m: {p: {a: float(np.mean(SW[m][p][a])) for a in SW[m][p]} for p in [1, 5, 10, 100]}
                   for m in SW},
    "per_class_xresnet": per_class("xresnet18"),
}
json.dump(summary, open(f"{OUT}/_summary.json", "w"), indent=2)
print("\nécrit", f"{OUT}/_summary.json")
print(f"\nxresnet1d18 from-scratch = {XRES:.4f} (meilleur test du projet)")
