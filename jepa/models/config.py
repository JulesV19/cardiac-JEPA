"""Hyperparamètres de modèle partagés par les deux encodeurs et le predictor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    grid_h: int = 12          # leads
    grid_w: int = 40          # patches temporels
    patch_len: int = 25       # échantillons par patch (0.25 s @ 100 Hz)
    embed_dim: int = 192      # ViT-tiny
    depth: int = 12
    heads: int = 3
    mlp_ratio: float = 4.0
    pred_dim: int = 96        # predictor plus étroit (bottleneck)
    pred_depth: int = 6
    pred_heads: int = 3

    encoder_type: str = "vit"           # "vit" (défaut), "cnn" ou "xresnet"
    # --- hyperparams encodeur CNN (ignorés si encoder_type != "cnn") ---
    cnn_channels: tuple = (64, 128)     # canaux de sortie par étage
    cnn_blocks_per_stage: int = 2       # blocs résiduels par étage
    cnn_strides: tuple = (5, 5)         # produit = downsample temporel (5*5=25 -> 1000/25=40)
    cnn_kernel: int = 7
    # --- hyperparams encodeur xresnet (ignorés si encoder_type != "xresnet") ---
    xr_arch: str = "xresnet1d18"        # xresnet1d18/34 (basic), 50/101 (bottleneck)
    xr_width_mult: float = 1.0          # scale toutes les largeurs (cale la capacité)
    xr_kernel: int = 5                  # kernel des convs (benchmark : 5)

    @property
    def num_tokens(self) -> int:
        return self.grid_h * self.grid_w


def build_tiny() -> ModelConfig:
    """Config ViT-tiny par défaut (voir plan)."""
    return ModelConfig()
