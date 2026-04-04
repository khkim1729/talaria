import torch
import torch.nn as nn

class RotationHead3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(2)
        flattened_dim = in_channels * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.classifier(x)