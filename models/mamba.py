"""
Missing-frame-aware Selective SSM (GapSSM / Mamba-inspired) model
adapted from AI_project/src/cardiac/models/gap_ssm.py for 1-channel (B, 1, T, H, W) clips.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class FrameEncoder(nn.Module):
    """2D per-frame encoder for 1-channel grey-scale echocardiography frames."""

    def __init__(self, width: int, kind: str = "resnet18") -> None:
        super().__init__()
        if kind == "resnet18":
            # GroupNorm avoids frame-count-dependent running statistics and supports batch size 1.
            net = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(8, channels))
            old_conv = net.conv1
            net.conv1 = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            net.fc = nn.Linear(net.fc.in_features, width)
            self.net = net
        elif kind == "tiny":
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, width),
            )
        else:
            raise ValueError(f"Unknown image encoder: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelectiveBlock(nn.Module):
    """Selective diagonal state-space block with continuous-time gap decay."""

    def __init__(
        self,
        width: int = 128,
        state_size: int = 16,
        dropout: float = 0.2,
        use_gap: bool = True,
        hold_missing: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.use_gap = use_gap
        self.hold_missing = hold_missing

        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * width)
        self.conv = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.parameters_proj = nn.Linear(width + 2, width + 2 * state_size)
        self.a_log = nn.Parameter(
            torch.arange(1, state_size + 1).float().log().repeat(width, 1)
        )
        self.gap_rate = nn.Parameter(torch.full((width, state_size), -4.0))
        self.skip = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        batch, length, width = features.shape
        u, gate = self.in_proj(self.norm(features)).chunk(2, -1)
        u = u * mask[..., None]
        u = F.silu(self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2))

        with torch.autocast(device_type=features.device.type, enabled=False):
            u, gate = u.float(), gate.float()
            A = -self.a_log.float().exp()
            h = torch.zeros(batch, width, self.state_size, device=features.device)
            last = torch.full((batch,), -1.0, device=features.device)
            outputs = []

            for t in range(length):
                present = mask[:, t].float()
                gap = (times[:, t].float() - last).clamp_min(0)
                if not self.use_gap:
                    gap = torch.ones_like(gap)
                meta = torch.stack([present, torch.log1p(gap)], -1)
                params = self.parameters_proj(torch.cat([u[:, t], meta], -1))
                dt_raw, B, C = torch.split(
                    params, [width, self.state_size, self.state_size], -1
                )
                dt = F.softplus(dt_raw).clamp(max=10)

                extra = (gap - 1).clamp_min(0)[:, None, None]
                if self.use_gap:
                    h_before = torch.exp(A[None] * F.softplus(self.gap_rate)[None] * extra) * h
                else:
                    h_before = h

                z = dt[..., None] * A[None]
                decay = torch.exp(z)
                integral = torch.expm1(z) / A[None]
                candidate = decay * h_before + integral * B[:, None, :] * u[:, t, :, None]

                if self.hold_missing:
                    h = torch.where(mask[:, t, None, None], candidate, h)
                else:
                    h = candidate

                last = torch.where(
                    mask[:, t] if self.hold_missing else torch.ones_like(mask[:, t]),
                    times[:, t].float(),
                    last,
                )
                y = (h * C[:, None, :]).sum(-1) + self.skip * u[:, t]
                outputs.append(y * F.silu(gate[:, t]))

            out = torch.stack(outputs, 1)

        return features + self.dropout(self.out_proj(out))


class GapSSMMamba(nn.Module):
    """
    Missing-frame-aware Temporal Mamba (GapSSM) model.
    Accepts (B, 1, T, H, W) grey-scale video clips and optional (mask, times),
    returning (B,) predicted EF.
    """

    def __init__(
        self,
        pretrained: bool = True,  # kept for uniform factory signature
        width: int = 128,
        state_size: int = 16,
        layers: int = 2,
        dropout: float = 0.2,
        encoder: str = "resnet18",
        use_mask: bool = True,
        use_gap: bool = True,
        hold_missing: bool = True,
    ) -> None:
        super().__init__()
        self.use_mask = use_mask
        self.encoder = FrameEncoder(width, encoder)
        self.blocks = nn.ModuleList(
            [
                SelectiveBlock(width, state_size, dropout, use_gap, hold_missing)
                for _ in range(layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, 1, T, H, W) -> transpose to (B, T, 1, H, W)
        video = x.transpose(1, 2)
        batch, length = video.shape[:2]

        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()

        if times is None:
            times = (
                torch.arange(length, device=x.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(batch, length)
            )

        effective_mask = mask if self.use_mask else torch.ones_like(mask)
        observed = video.flatten(0, 1)[effective_mask.flatten()]
        values = self.encoder(observed)

        features = torch.zeros(
            batch * length, values.shape[-1], device=x.device, dtype=values.dtype
        )
        features[effective_mask.flatten()] = values
        features = features.reshape(batch, length, -1)

        for block in self.blocks:
            features = block(features, effective_mask, times)

        pooled = (features * effective_mask[..., None]).sum(1) / effective_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        return self.head(pooled).squeeze(-1)
