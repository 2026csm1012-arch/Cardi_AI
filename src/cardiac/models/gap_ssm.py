"""Custom Mamba-inspired selective diagonal SSM. No claim of official Mamba equivalence."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class FrameEncoder(nn.Module):
    def __init__(self, width, kind='resnet18'):
        super().__init__()
        if kind == 'resnet18':
            # GroupNorm avoids frame-count-dependent running statistics and supports batch size one.
            net = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(8, channels))
            net.fc = nn.Linear(net.fc.in_features, width)
            self.net = net
        elif kind == 'tiny':
            self.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, width))
        else:
            raise ValueError('Unknown image encoder.')

    def forward(self, x):
        return self.net(x)


class SelectiveBlock(nn.Module):
    def __init__(self, width=128, state_size=16, dropout=.2, use_gap=True, hold_missing=True):
        super().__init__()
        self.width, self.state_size = width, state_size
        self.use_gap, self.hold_missing = use_gap, hold_missing
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * width)
        self.conv = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.parameters_proj = nn.Linear(width + 2, width + 2 * state_size)
        self.a_log = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(width, 1))
        self.gap_rate = nn.Parameter(torch.full((width, state_size), -4.0))
        self.skip = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, mask, times):
        batch, length, width = features.shape
        u, gate = self.in_proj(self.norm(features)).chunk(2, -1)
        # Missing pixels cannot enter the causal convolution through projection bias.
        u = u * mask[..., None]
        u = F.silu(self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2))
        # State math stays float32 even when image convolutions use autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            u, gate = u.float(), gate.float()
            A = -self.a_log.float().exp()
            h = torch.zeros(batch, width, self.state_size, device=features.device)
            last = torch.full((batch,), -1., device=features.device)
            outputs = []
            for t in range(length):
                present = mask[:, t].float()
                gap = (times[:, t].float() - last).clamp_min(0)
                if not self.use_gap:
                    gap = torch.ones_like(gap)
                meta = torch.stack([present, torch.log1p(gap)], -1)
                # Cast inputs to weight dtype only if the entire model was explicitly cast.
                params = self.parameters_proj(torch.cat([u[:, t], meta], -1))
                dt_raw, B, C = torch.split(params, [width, self.state_size, self.state_size], -1)
                dt = F.softplus(dt_raw).clamp(max=10)
                # Learned gap decay acts once, when the next observation arrives.
                extra = (gap - 1).clamp_min(0)[:, None, None]
                if self.use_gap:
                    h_before = torch.exp(A[None] * F.softplus(self.gap_rate)[None] * extra) * h
                else:
                    h_before = h
                z = dt[..., None] * A[None]
                decay = torch.exp(z)
                # Exact zero-order-hold input integral for a diagonal negative A.
                integral = torch.expm1(z) / A[None]
                candidate = decay * h_before + integral * B[:, None, :] * u[:, t, :, None]
                if self.hold_missing:
                    h = torch.where(mask[:, t, None, None], candidate, h)
                else:
                    h = candidate
                last = torch.where(mask[:, t] if self.hold_missing else torch.ones_like(mask[:, t]),
                                   times[:, t].float(), last)
                y = (h * C[:, None, :]).sum(-1) + self.skip * u[:, t]
                outputs.append(y * F.silu(gate[:, t]))
            out = torch.stack(outputs, 1)
        return features + self.dropout(self.out_proj(out))


class GapSSM(nn.Module):
    def __init__(self, width=128, state_size=16, layers=2, dropout=.2,
                 encoder='resnet18', use_mask=True, use_gap=True, hold_missing=True):
        super().__init__()
        self.use_mask = use_mask
        self.encoder = FrameEncoder(width, encoder)
        self.blocks = nn.ModuleList([SelectiveBlock(width, state_size, dropout, use_gap, hold_missing)
                                     for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, video, mask, times):
        if mask.ndim != 2 or mask.shape != video.shape[:2] or times.shape != mask.shape:
            raise ValueError('Mask and timestamps must have shape B,T.')
        if not mask.any(dim=1).all():
            raise ValueError('Every clip needs at least one observed frame.')
        if (times[:, 1:] < times[:, :-1]).any():
            raise ValueError('Timestamps must not decrease.')
        effective_mask = mask if self.use_mask else torch.ones_like(mask)
        batch, length = video.shape[:2]
        observed = video.flatten(0, 1)[effective_mask.flatten()]
        values = self.encoder(observed)
        features = torch.zeros(batch * length, values.shape[-1], device=video.device, dtype=values.dtype)
        features[effective_mask.flatten()] = values
        features = features.reshape(batch, length, -1)
        for block in self.blocks:
            features = block(features, effective_mask, times)
        pooled = (features * effective_mask[..., None]).sum(1) / effective_mask.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)
