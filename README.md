# Cardiac JEPA — auto-supervision par prédiction latente sur ECG (PTB-XL)

Un **JEPA** (Joint-Embedding Predictive Architecture) qui prédit les *embeddings* de zones
masquées d'un ECG puis un classifieur d'anomalies par-dessus. On mesure ce
que le pré-entraînement apporte, et on le confronte à des baselines supervisées à capacité égale.

**Dataset.** PTB-XL — 21 837 ECG, 12 dérivations, 10 s, 100 Hz, 5 super-classes (NORM, MI, STTC, CD,
HYP), multi-label. **Métrique : macro-AUROC.** Splits standards : pré-entraînement folds 1-8,
sélection d'epoch sur fold 9, **test sur le fold 10** (jamais vu). Tous les chiffres ci-dessous sont
en **test / fold 10**.

---

## Résultats en bref

1. **Le JEPA apprend de vraies représentations.** Sur encodeur **gelé**, une sonde linéaire passe de
   0,78 (encodeur aléatoire) à **0,84** — soit **+0,06 à +0,08** sans aucun label pendant le
   pré-entraînement. Pas de *collapse*.
2. **Mais le fine-tuning complet efface ce gain.** À 100 % des labels, l'écart JEPA − aléatoire tombe
   à **~0** (+0,004 à +0,009). La valeur du SSL est **en régime peu-de-labels**, où elle est
   statistiquement établie (5 graines à 1 % / 5 %).
3. **Le levier dominant à pleine supervision, c'est le *design* du backbone — pas la taille.** À
   capacité égale, un **xresnet1d18 from-scratch (0,894)** fait aussi bien ou mieux que le meilleur
   CNN pré-entraîné JEPA (0,888) : le design (ResNet-D) relève le plafond ~0,88 → ~0,90, tandis
   qu'agrandir un CNN de 1,27 M à 6,25 M **ne le monte pas**. Le frein restant est la
   **généralisation** sur 17 k ECG, pas la capacité. *(Réserve : l'écart xresnet↔CNN-JEPA n'est que
   de +0,005 à une seule graine, dans le bruit plausible ; et le xresnet n'a pas été pré-entraîné, donc
   backbone et SSL ne sont pas comparés à isopérimètre.)*

---

## 1 · Le pré-entraînement JEPA est sain

![Pré-entraînement](figures/5_pretraining.png)

L'encodeur **ne s'effondre pas** : le rang effectif monte et se stabilise haut (les modèles utilisent
la majeure partie de leurs dimensions). Le **prédicteur**, lui, retombe dans une **régression vers la
moyenne** (le cosinus prédiction↔cible culmine tôt puis décline) — un plateau *intrinsèque au
prédicteur sous incertitude*, identique sur ViT et CNN, et **sans conséquence** : la détection lit les
embeddings de l'encodeur (sains), jamais la prédiction.

## 2 · Sur features gelées, le SSL fonctionne

![Sonde gelée](figures/2_probe.png)

Encodeur gelé, sonde linéaire entraînée par-dessus. Le pré-entraînement apporte un gain net et
reproductible sur les deux familles de backbone.

| sonde linéaire (gelé) | aléatoire | JEPA | écart |
|---|---:|---:|---:|
| ViT-tiny | 0,7825 | **0,8437** | **+0,061** |
| CNN v2 | 0,7623 | **0,8413** | **+0,079** |

## 3 · Peu-de-labels : là où le SSL prouve sa valeur

![Peu de labels](figures/3_lowlabel.png)

À 1 % des labels (~171 ECG), le pré-entraînement fait gagner jusqu'à **+0,066** (ViT) ; l'écart
rétrécit à mesure que les labels arrivent. Les écarts à 1 % et 5 % sont **pairés par graine sur 5
graines** et **excluent zéro** (t ≈ 6 à 1 %, t ≈ 14–21 à 5 %) — c'est un effet, pas du bruit.
Efficacité-label ~2 à 5× : JEPA à 1 % ≈ aléatoire à 5 %.

![Le gain SSL s'efface](figures/4_ssl_gain.png)

Même **forme monotone décroissante** pour les trois modèles, jusqu'à ~0 à 100 %. Le gain SSL est
**systématiquement plus petit sur CNN** que sur ViT : le bon biais inductif du CNN capte déjà une
partie de ce que l'auto-supervision apportait au transformer.

| écart JEPA − aléatoire | 1 % | 5 % | 10 % | 100 % |
|---|---:|---:|---:|---:|
| ViT | +0,066 | +0,037 | +0,025 | +0,003 |
| CNN v1 | +0,038 | +0,029 | +0,021 | +0,007 |
| CNN v2 | +0,052 | +0,025 | +0,015 | +0,004 |

## 4 · Le design du backbone relève le plafond

![Backbone](figures/1_backbone.png)

À 100 % des labels (fine-tuné), le **CNN bat le ViT** des deux côtés (aléatoire *et* JEPA, écart
~+0,01–0,02), et un **xresnet1d18 from-scratch atteint 0,894** — le meilleur score du projet, **sans
aucun pré-entraînement** (mais à +0,005 du meilleur CNN-JEPA, sur une seule graine : au niveau, pas une
domination établie). Deux enseignements francs :

- **La capacité n'est pas le levier.** Le CNN v2 (6,25 M) ne bat pas le CNN v1 (1,27 M) au plafond.
- **Le *design* du backbone, si.** xresnet (ResNet-D, stem 3-convs) relève le plafond ~0,88 → ~0,90 à
  capacité égale.

| plafond 100 % (fine-tuné) | params | aléatoire | JEPA |
|---|---:|---:|---:|
| ViT-tiny | 6,0 M | 0,8657 | 0,8748 |
| CNN v1 | 1,27 M | 0,8817 | **0,8884** |
| CNN v2 | 6,25 M | 0,8789 | 0,8826 |
| **xresnet1d18** *(sans SSL)* | 6,30 M | **0,8936** | — |

## 5 · Le mur est la généralisation, pas la capacité

![Généralisation](figures/6_generalization.png)

En fine-tuning, la val-AUROC **pique vers l'epoch 6-10 puis décline** pendant que la train-loss tend
vers 0 : **sur-apprentissage** franc sur 17 k ECG. Preuve mécanistique : le petit **CNN v1 est
incapable de fitter le train** (loss bloquée ~0,18) et atteint pourtant **le même pic** que le CNN v2
qui, lui, **mémorise** le train (loss ~0,015). Fitter davantage le train ne monte pas le plafond —
c'est la généralisation qui bloque, pas les paramètres.

---

## Récapitulatif complet (macro-AUROC test)

| régime | ViT (6 M) | CNN v1 (1,27 M) | CNN v2 (6,25 M) | xresnet (6,30 M) |
|---|---:|---:|---:|---:|
| Sonde linéaire (gelé) | 0,8437 | 0,8167 | 0,8413 | — |
| Fine-tuné 1 % | 0,7798 | 0,7780 | 0,8109 | — |
| Fine-tuné 5 % | 0,8170 | 0,8241 | 0,8457 | — |
| Fine-tuné 10 % | 0,8317 | 0,8527 | 0,8584 | — |
| Fine-tuné 100 % | 0,8748 | **0,8884** | 0,8826 | **0,8936** |

*(colonnes ViT/CNN : bras JEPA pré-entraîné ; xresnet : supervisé from-scratch.)*

## Positionnement honnête vs littérature

- La **SOTA supervisée** sur PTB-XL superdiagnostic (~0,93 ; resnet1d_wang, xresnet1d101) s'entraîne
  **et** teste sur **les mêmes splits que nous**. L'écart nous↔0,93 est donc **archi + recette**
  (augmentation de données, design CNN dédié, résolution 500 Hz ?), **pas les données**.
- Les papiers **JEPA-ECG** (~0,92 linéaire) **pré-entraînent sur des corpus externes >100 k–1 M ECG**
  puis PTB-XL — **non comparables** à notre SSL entraîné sur les **17 k de PTB-XL seulement**. Notre
  +0,06 en sonde depuis 17 k-seulement est en ce sens un résultat solide.

## Limites

- AUROC de rang **uniquement** — pas de seuils ni de calibration.
- Barres d'erreur multi-graines seulement à 1 % et 5 % ; **10 % et 100 % à une seule graine** — les
  écarts fins au plafond (dont xresnet +0,005 vs CNN-JEPA) ne sont donc **pas** statistiquement établis.
- Backbone et SSL **non croisés** : le xresnet n'a jamais été pré-entraîné en JEPA. On mesure l'effet
  *design de backbone* et l'effet *SSL* sur des backbones différents, pas l'un contre l'autre.
- Le xresnet est mesuré à sa taille iso-capacité (6,30 M), pas au xresnet1d101 plein (33 M, qui
  mémoriserait ~17 k échantillons) : on teste le **design**, pas la reproduction du 0,93.

---

## Reproduire

```bash
# Pré-entraînement JEPA (ViT ou CNN)
python -m jepa.train --config jepa/configs/tiny.yaml       # ViT
python -m jepa.train --config jepa/configs/cnn_v2.yaml     # CNN

# Sonde linéaire (encodeur gelé) et fine-tuning (--random-init pour la baseline iso-archi)
python -m jepa.probe    --ckpt runs/cnn_v2/best.pt
python -m jepa.classify --ckpt runs/cnn_v2/best.pt --train-frac 0.05 --seed 0

# Baseline supervisée from-scratch (xresnet)
python -m jepa.supervised --config jepa/configs/xresnet.yaml --out runs/xresnet18

# Régénérer les figures de ce README depuis runs/
python make_figures.py
```

**Structure.** `jepa/models/` (config, ViT, CNN, predictor) · `jepa/train/` · `jepa/probe/` ·
`jepa/classify/` · `jepa/supervised/` (xresnet) · `jepa/decode/` · `jepa/configs/*.yaml`.
Le rapport détaillé de la campagne CNN vit dans [`rapport/RAPPORT_CNN_JEPA.md`](rapport/RAPPORT_CNN_JEPA.md).
