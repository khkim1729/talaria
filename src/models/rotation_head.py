import torch.nn as nn

class RotationHead3D(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        # 3D 볼륨(D, H, W) 특징을 (1, 1, 1)로 압축하여 이미지 전체의 대표 정보를 뽑아냅니다.
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # 압축된 특징을 바탕으로 4개의 클래스(각도)로 분류합니다.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # 과적합 방지
            nn.Linear(in_channels // 2, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)