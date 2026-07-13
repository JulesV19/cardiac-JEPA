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

    encoder_type: str = "vit"           # "vit" (défaut) ou "cnn"
    # --- hyperparams encodeur CNN (ignorés si encoder_type == "vit") ---
    cnn_channels: tuple = (64, 128)     # canaux de sortie par étage
    cnn_blocks_per_stage: int = 2       # blocs résiduels par étage
    cnn_strides: tuple = (5, 5)         # produit = downsample temporel (5*5=25 -> 1000/25=40)
    cnn_kernel: int = 7

    @property
    def num_tokens(self) -> int:
        return self.grid_h * self.grid_w


def build_tiny() -> ModelConfig:
    """Config ViT-tiny par défaut (voir plan)."""
    return ModelConfig()
