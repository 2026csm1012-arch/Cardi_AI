import torch
from torch import nn
from torchvision.models.video import r3d_18, r2plus1d_18, R3D_18_Weights, R2Plus1D_18_Weights


class VideoBaseline(nn.Module):
    def __init__(self, architecture='r3d_18', dropout=.2, pretrained=False):
        super().__init__()
        if architecture == 'r3d_18':
            self.network = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
        elif architecture == 'r2plus1d_18':
            self.network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError('Baseline must be r3d_18 or r2plus1d_18.')
        self.network.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.network.fc.in_features, 1))

    def forward(self, video, mask=None, times=None):
        # Inputs are B,T,C,H,W. The baseline receives the same reconstructed pixels.
        return self.network(video.transpose(1, 2)).squeeze(-1)
