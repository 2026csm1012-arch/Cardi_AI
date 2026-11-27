"""
Model registry for Cardi_AI.
Switch models via config.model_name or --model on the CLI.
"""

import torch.nn as nn

from models.r2plus1d import R2Plus1D18Baseline
from models.mamba import GapSSMMamba

MODEL_REGISTRY = {
    "r2plus1d_18": R2Plus1D18Baseline,
    "mamba": GapSSMMamba,
    "gap_ssm": GapSSMMamba,
}


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[key](pretrained=pretrained)
