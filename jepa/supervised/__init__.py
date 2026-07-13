"""Baseline supervisée xresnet1d101 (point de référence backbone, sans JEPA)."""
from .model import XResNet1d, make_xresnet

__all__ = ["XResNet1d", "make_xresnet"]
