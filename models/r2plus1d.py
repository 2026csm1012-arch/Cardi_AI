"""
R(2+1)D-18 baseline video regression model for 1-channel grey-scale clips.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1D18Baseline(nn.Module):
    """
    R(2+1)D-18 baseline video regression network.
    Accepts (B, 1, 32, 112, 112) and outputs (B,) predicted EF.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = R2Plus1D_18_Weights.DEFAULT
            except Exception:
                weights = None

        try:
            self.backbone = r2plus1d_18(weights=weights)
        except Exception as exc:
            print(f"[Warning] Could not load pretrained weights ({exc}); using random init.")
            self.backbone = r2plus1d_18(weights=None)

        # Adapt stem[0] from 3-channel RGB to 1-channel grey-scale
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained and weights is not None:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        self.backbone.stem[0] = new_conv

        # Replace classification head with scalar EF regression head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 112, 112) -> returns (B,)
        return self.backbone(x).squeeze(-1)
