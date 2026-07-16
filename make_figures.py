#!/usr/bin/env python3
"""Génère les figures du README depuis runs/results.json (+ les metrics.csv).

Source unique = l'agrégat produit par `python -m jepa.aggregate` (moyennes ± écart-type
inter-graines, écarts pairés JEPA−scratch + t de Student). Aucun chiffre codé en dur.
Barres d'erreur = écart-type inter-graines. Défensif : saute proprement ce qui manque.

    python make_figures.py
"""
import json
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = "runs"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---- palette CVD-safe ----
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARM = {"scratch": "#eb6834", "jepa": "#2a78d6"}
BBCOL = {"vit": "#4a3aa7", "cnn": "#199e70", "xresnet": "#c81e5a"}
EMBED = {"vit": 192, "cnn": 256, "xresnet": 256}
BBNAME = {"vit": "ViT-tiny", "cnn": "CNN", "xresnet": "xresnet"}
BACKBONES = ["vit", "cnn", "xresnet"]
FRACS = [("ft1", 1), ("ft5", 5), ("ft10", 10), ("ft100", 100)]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK2, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 150,
})

RES = json.load(open(f"{RUNS}/results.json"))
CELLS, GAPS = RES["cells"], RES["paired_gaps"]


def cell(bb, arm, reg):
    return CELLS.get(f"{bb}|{arm}|{reg}")


def val(bb, arm, reg, metric="auroc"):
    """(mean, sd) inter-graines, ou (None, None) si absent."""
    c = cell(bb, arm, reg)
    return (c[metric]["mean"], c[metric]["sd"]) if c else (None, None)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("écrit", f"{OUT}/{name}.png")


# =========================================================================
# FIG 1 — Plafond 100 % : AUROC + AUPRC, scratch vs JEPA + xresnet BN supervisé
# =========================================================================
def _ceiling(ax, metric, title):
    x = np.arange(len(BACKBONES))
    w = 0.34
    for k, arm in enumerate(("scratch", "jepa")):
        means = [val(bb, arm, "ft100", metric)[0] for bb in BACKBONES]
        sds = [val(bb, arm, "ft100", metric)[1] for bb in BACKBONES]
        ax.bar(x + (k - 0.5) * w, [m or 0 for m in means], w,
               yerr=[s or 0 for s in sds], capsize=3, color=ARM[arm],
               label="aléatoire" if arm == "scratch" else "JEPA")
    # référence xresnet BN supervisé (hors grille SSL)
    sup = val("xresnet_bn", "supervised", "ft100", metric)
    if sup[0] is not None:
        ax.axhline(sup[0], color=INK, ls="--", lw=1.2)
        ax.text(len(BACKBONES) - 0.5, sup[0], f" xresnet BN {sup[0]:.3f}",
                color=INK, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([BBNAME[b] for b in BACKBONES])
    ax.set_ylabel(f"macro-{metric.upper()} (test)")
    ax.set_title(title)
    lo = min([v for b in BACKBONES for a in ("scratch", "jepa")
              for v in [val(b, a, "ft100", metric)[0]] if v is not None] or [0.5])
    ax.set_ylim(max(0, lo - 0.03), None)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
_ceiling(a1, "auroc", "macro-AUROC")
_ceiling(a2, "auprc", "macro-AUPRC")
a1.legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("Plafond à 100 % des labels (fine-tuné, 5 graines ± écart-type)",
             fontsize=13, fontweight="bold", color=INK, y=1.03)
finish(fig, "1_backbone")

# =========================================================================
# FIG 2 — Sonde gelée : SSL sur les représentations (scratch vs JEPA)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 3.4))
present = [b for b in BACKBONES if cell(b, "jepa", "probe")]
y = np.arange(len(present))
for i, bb in enumerate(present):
    r = val(bb, "scratch", "probe")[0]
    j = val(bb, "jepa", "probe")[0]
    if r is None or j is None:
        continue
    ax.plot([r, j], [i, i], color=MUTED, lw=2, zorder=1)
    ax.scatter([r], [i], s=90, color=ARM["scratch"], zorder=3)
    ax.scatter([j], [i], s=90, color=ARM["jepa"], zorder=3)
    ax.annotate(f"{r:.3f}", (r, i), textcoords="offset points", xytext=(0, -15),
                ha="center", color=ARM["scratch"], fontsize=9)
    ax.annotate(f"{j:.3f}", (j, i), textcoords="offset points", xytext=(0, 9),
                ha="center", color=ARM["jepa"], fontsize=9, fontweight="bold")
    ax.annotate(f"+{j-r:.3f}", (j, i), textcoords="offset points", xytext=(30, 0),
                va="center", color=INK, fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([BBNAME[b] for b in present])
ax.set_xlabel("macro-AUROC (test) — sonde linéaire sur encodeur gelé")
ax.set_title("Sonde linéaire sur encodeur gelé")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=ARM["scratch"], label="aléatoire"),
                   Line2D([0], [0], marker="o", ls="", color=ARM["jepa"], label="JEPA")],
          loc="lower right", frameon=False, fontsize=9)
finish(fig, "2_probe")

# =========================================================================
# FIG 3 — Peu-de-labels : AUROC vs %labels, 3 backbones (couleur) × arm (style) ± sd
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 4.8))
pcts = [p for _, p in FRACS]
for bb in BACKBONES:
    for arm, ls in (("jepa", "-"), ("scratch", "--")):
        ms = [val(bb, arm, reg)[0] for reg, _ in FRACS]
        ss = [val(bb, arm, reg)[1] for reg, _ in FRACS]
        if any(m is None for m in ms):
            continue
        ax.errorbar(pcts, ms, yerr=[s or 0 for s in ss], marker="o", ms=5, lw=2, ls=ls,
                    capsize=2.5, color=BBCOL[bb])
ax.set_xscale("log"); ax.set_xticks(pcts); ax.set_xticklabels([f"{p}%" for p in pcts])
ax.set_xlabel("part des labels d'entraînement (échelle log)")
ax.set_ylabel("macro-AUROC (test)")
ax.set_title("macro-AUROC test vs part des labels — 3 backbones")
ax.set_xlim(0.8, 160)
bb_h = [Line2D([0], [0], color=BBCOL[b], lw=2, label=BBNAME[b]) for b in BACKBONES]
st_h = [Line2D([0], [0], color=INK2, lw=2, ls="-", label="JEPA"),
        Line2D([0], [0], color=INK2, lw=2, ls="--", label="aléatoire")]
ax.legend(handles=bb_h + st_h, frameon=False, fontsize=9, loc="lower right", ncol=2)
finish(fig, "3_lowlabel")

# =========================================================================
# FIG 4 — Écart pairé JEPA − scratch vs %labels, 3 backbones (± sd, t annoté)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for bb in BACKBONES:
    xs, ys, es = [], [], []
    for reg, p in FRACS:
        g = GAPS.get(f"{bb}|{reg}")
        if g:
            xs.append(p); ys.append(g["mean"]); es.append(g["sd"])
    if not xs:
        continue
    ax.errorbar(xs, ys, yerr=es, marker="o", ms=6, lw=2, capsize=3,
                color=BBCOL[bb], label=BBNAME[bb])
ax.axhline(0, color=MUTED, lw=1, ls="--")
ax.set_xscale("log"); ax.set_xticks([p for _, p in FRACS])
ax.set_xticklabels([f"{p}%" for _, p in FRACS])
ax.set_xlabel("part des labels d'entraînement (échelle log)")
ax.set_ylabel("écart JEPA − aléatoire (macro-AUROC)")
ax.set_title("Écart pairé JEPA − aléatoire vs part des labels")
ax.set_xlim(0.8, 160)
ax.legend(frameon=False, loc="upper right")
finish(fig, "4_ssl_gain")

# =========================================================================
# FIG 5 — Diagnostics anti-collapse : VICReg (var, cov) + rang + std + cos + pred/tgt
# =========================================================================
PRE = {}
for bb in BACKBONES:
    p = f"{RUNS}/{bb}/pretrain/metrics.csv"
    if os.path.exists(p):
        PRE[bb] = pd.read_csv(p)

fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2))
(ax_var, ax_cov, ax_cos), (ax_std, ax_rank, ax_pred) = axs

for bb, df in PRE.items():
    tr, va = df[df.phase == "train"], df[df.phase == "val"]
    c = BBCOL[bb]
    ax_var.plot(tr.step, tr["var"], color=c, lw=2, label=BBNAME[bb])
    ax_cov.plot(tr.step, tr["cov"], color=c, lw=2)
    ax_cos.plot(va.epoch, va.cos, color=c, lw=2)
    ax_std.plot(va.epoch, va.emb_std_ctx, color=c, lw=2, ls="-")
    ax_std.plot(va.epoch, va.emb_std_tgt, color=c, lw=1.5, ls="--")
    ax_rank.plot(va.epoch, va.eff_rank_ctx / EMBED[bb], color=c, lw=2, ls="-")
    ax_rank.plot(va.epoch, va.eff_rank_tgt / EMBED[bb], color=c, lw=1.5, ls="--")
    ax_pred.plot(va.epoch, va.pred_std, color=c, lw=2, ls="-")
    ax_pred.plot(va.epoch, va.emb_std_tgt, color=c, lw=1.5, ls="--")

ax_var.set_title("VICReg — terme de variance")
ax_var.set_xlabel("step"); ax_var.set_ylabel("var  (hinge max(0, 1−std))")
ax_var.axhline(0, color=MUTED, lw=1, ls=":")

ax_cov.set_title("VICReg — terme de covariance")
ax_cov.set_xlabel("step"); ax_cov.set_ylabel("cov  (redondance)")

ax_cos.set_title("cos(prédiction, cible)")
ax_cos.set_xlabel("epoch"); ax_cos.set_ylabel("cos")

ax_std.set_title("Écart-type des embeddings")
ax_std.set_xlabel("epoch"); ax_std.set_ylabel("std moyen par dim")
ax_std.axhline(1.0, color=MUTED, lw=1, ls=":")

ax_rank.set_title("Rang effectif normalisé")
ax_rank.set_xlabel("epoch"); ax_rank.set_ylabel("rang effectif / dim"); ax_rank.set_ylim(0, 1)

ax_pred.set_title("Écart-type : prédiction vs cible")
ax_pred.set_xlabel("epoch"); ax_pred.set_ylabel("std moyen par dim")

# légendes : couleurs backbones (une fois) + clé style contexte/cible
ax_var.legend(frameon=False, loc="upper right", fontsize=9)
style = [Line2D([0], [0], color=INK2, lw=2, ls="-", label="contexte / prédiction"),
         Line2D([0], [0], color=INK2, lw=1.5, ls="--", label="cible")]
ax_rank.legend(handles=style, frameon=False, loc="lower right", fontsize=8)
fig.suptitle("Pré-entraînement JEPA — diagnostics anti-collapse (suivi fold 9)",
             fontsize=13, fontweight="bold", color=INK, y=1.0)
finish(fig, "5_pretraining")

# =========================================================================
# FIG 5b — Progression du pré-entraînement : loss JEPA + sonde macro-AUROC in-loop
# =========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for bb, df in PRE.items():
    tr, va = df[df.phase == "train"], df[df.phase == "val"]
    a1.plot(tr.step, tr["total"], color=BBCOL[bb], lw=2, label=BBNAME[bb])
    pa = va.dropna(subset=["probe_auroc"])
    a2.plot(pa.epoch, pa["probe_auroc"], color=BBCOL[bb], lw=2, label=BBNAME[bb])
a1.set_title("Loss d'entraînement JEPA")
a1.set_xlabel("step"); a1.set_ylabel("loss totale")
a2.set_title("Sonde macro-AUROC (in-loop, fold 9)")
a2.set_xlabel("epoch"); a2.set_ylabel("macro-AUROC")
a2.legend(frameon=False, loc="lower right")
fig.suptitle("Pré-entraînement JEPA — loss et AUROC mesurées pendant l'entraînement",
             fontsize=13, fontweight="bold", color=INK, y=1.02)
finish(fig, "5b_pretrain_progress")

# =========================================================================
# FIG 6 — Généralisation : val AUROC + train loss vs epoch (ft100, graine 0)
# =========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
any_ft = False
for bb in BACKBONES:
    p = f"{RUNS}/{bb}/jepa/ft100/s0/metrics.csv"
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    any_ft = True
    a1.plot(df.epoch, df.val_macro_auroc, color=BBCOL[bb], lw=2, label=BBNAME[bb])
    be = int(df.val_macro_auroc.idxmax())
    a1.scatter([df.epoch.iloc[be]], [df.val_macro_auroc.iloc[be]], s=80,
               color=BBCOL[bb], edgecolor=SURF, linewidth=1.5, zorder=5)
    a2.plot(df.epoch, df.train_loss, color=BBCOL[bb], lw=2, label=BBNAME[bb])
a1.set_title("val macro-AUROC (point = meilleur epoch)")
a1.set_xlabel("epoch de fine-tuning"); a1.set_ylabel("val macro-AUROC")
a2.set_title("train loss")
a2.set_xlabel("epoch de fine-tuning"); a2.set_ylabel("train loss")
if any_ft:
    a1.legend(frameon=False, loc="lower right", fontsize=9)
fig.suptitle("Fine-tuning 100 % — courbes d'entraînement (graine 0)",
             fontsize=13, fontweight="bold", color=INK, y=1.03)
finish(fig, "6_generalization")

# =========================================================================
# FIG 7 — F1 vs seuil de décision : chaque backbone comparé à SA version from-scratch
#   (iso-architecture), à 1 % et 5 % des labels. Bandes = moyenne ± écart-type sur les
#   graines. Couleur = régime (1 % / 5 %), trait plein = JEPA / tireté = supervisé.
#   UNE figure par backbone. Défensif : backbone sauté si ses thresholds.json manquent.
# =========================================================================
C1, C5 = "#d1495b", "#2a78d6"                     # 1 % (rouge), 5 % (bleu)


def _agg_seeds(dirrel):
    """Agrège toutes les graines d'un modèle : moyenne + écart-type des courbes par τ."""
    files = sorted(glob.glob(f"{RUNS}/{dirrel}/s*/thresholds.json"))
    ms = [json.load(open(f)) for f in files]
    if not ms:
        return None
    panels = [c for c in ms[0]["classes"] if c in ms[0]["curves"]] + ["macro"]
    out = {"taus": ms[0]["thresholds"], "panels": panels, "n": len(ms),
           "mean": {}, "sd": {}, "prev": {}}
    for name in panels:
        out["mean"][name], out["sd"][name] = {}, {}
        prevs = [m["curves"].get(name, {}).get("prevalence") for m in ms]
        prevs = [p for p in prevs if p is not None]
        out["prev"][name] = float(np.mean(prevs)) if prevs else None
        for metric in ("sensitivity", "specificity", "precision", "f1"):
            stack = [m["curves"][name][metric] for m in ms
                     if m["curves"].get(name) and metric in m["curves"][name]]
            if not stack:
                continue
            arr = np.array(stack, dtype=float)          # (graines, τ)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["mean"][name][metric] = np.nanmean(arr, axis=0)
                out["sd"][name][metric] = np.nanstd(arr, axis=0)
    return out


def _seuil_sel(bb):
    """(agg, couleur, style) pour JEPA/scratch × 1%/5% du backbone bb, si dispo."""
    spec = [(f"{bb}/jepa/ft1", C1, "-"), (f"{bb}/scratch/ft1", C1, "--"),
            (f"{bb}/jepa/ft5", C5, "-"), (f"{bb}/scratch/ft5", C5, "--")]
    return [(a, col, ls) for dirrel, col, ls in spec
            if (a := _agg_seeds(dirrel)) is not None]


for bb in BACKBONES:
    SEL = _seuil_sel(bb)
    if not SEL:
        print(f"(FIG 7 {bb} sautée : thresholds.json 1%/5% absents.)")
        continue
    panels = SEL[0][0]["panels"]
    nseed = max(a["n"] for a, *_ in SEL)
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
    for k, name in enumerate(panels):
        ax = axs[k // ncol][k % ncol]
        for a, col, ls in SEL:
            mu = a["mean"].get(name, {}).get("f1")
            sd = a["sd"].get(name, {}).get("f1")
            if mu is None:
                continue
            taus = np.array(a["taus"])
            if sd is not None and a["n"] > 1:           # bande ± écart-type inter-graines
                ax.fill_between(taus, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.plot(taus, mu, color=col, ls=ls, lw=2.0, alpha=0.98)
        prev = SEL[0][0]["prev"].get(name)
        ax.set_title(name + (f"  (prév. {prev:.0%})" if prev is not None else ""),
                     fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("seuil de décision τ")
        if k % ncol == 0:
            ax.set_ylabel("F1")
    for k in range(len(panels), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    reg_h = [Line2D([0], [0], color=C1, lw=2.4, label="1 % des labels"),
             Line2D([0], [0], color=C5, lw=2.4, label="5 % des labels")]
    arm_h = [Line2D([0], [0], color=INK2, lw=2, ls="-", label="JEPA (pré-entraîné)"),
             Line2D([0], [0], color=INK2, lw=2, ls="--", label="supervisé (scratch)")]
    axs[0][0].legend(handles=reg_h, frameon=False, fontsize=8, loc="lower left")
    axs[0][-1].legend(handles=arm_h, frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"F1 selon le seuil τ — {BBNAME[bb]} : JEPA vs supervisé (scratch) en "
                 f"peu-de-labels (test, moyenne ± écart-type, {nseed} graines)",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    finish(fig, f"7_seuil_f1_{bb}")

print("\nFigures générées dans", OUT)
