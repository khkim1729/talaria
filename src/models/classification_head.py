"""
TALARIA Classification Head.

deep_feat → Global Average Pooling → T-stage / N-stage classification

T-stage: T1 / T2 / T3 / T4  (4 classes)
N-stage: N0 / N1             (2 classes)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional


class ClassificationHead(nn.Module):
    """
    Dual classification head for T-stage and N-stage.

    Args:
        in_ch:      encoder deep_feat channels (320)
        t_classes:  number of T-stage classes (4)
        n_classes:  number of N-stage classes (2)
        dropout:    dropout rate
    """

    def __init__(
        self,
        in_ch: int = 320,
        t_classes: int = 4,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.t_head = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, t_classes),
        )

        self.n_head = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        deep_feat: torch.Tensor,
        apply_manifold_mixup: bool = False,
        mixup_alpha: float = 2.0,
        mixup_prob: float = 1.0,
        perm_idx: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ):
        """
        Args:
            deep_feat: (B, 320, D/16, H/16, W/16)
        Returns:
            t_logit: (B, t_classes)
            n_logit: (B, n_classes)
            mixup_meta: dict containing
                - mixup_lam (Optional[float])
                - mixup_perm (Optional[torch.Tensor])
                - mixup_applied (bool)
        """
        x = self.gap(deep_feat).flatten(1)  # (B, 320)
        mixup_applied = False
        mixup_lam: Optional[float] = None
        mixup_perm: Optional[torch.Tensor] = None

        if self.training and apply_manifold_mixup and torch.rand((), device=x.device) < mixup_prob:
            bsz = x.size(0)
            mixup_perm = perm_idx if perm_idx is not None else torch.randperm(bsz, device=x.device)

            if lam is None:
                beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
                mixup_lam = float(beta.sample().item())
            else:
                mixup_lam = float(lam)

            x = mixup_lam * x + (1.0 - mixup_lam) * x[mixup_perm]
            mixup_applied = True

        mixup_meta: Dict[str, Any] = {
            'mixup_lam': mixup_lam,
            'mixup_perm': mixup_perm,
            'mixup_applied': mixup_applied,
        }

        return self.t_head(x), self.n_head(x), mixup_meta


if __name__ == '__main__':
    head = ClassificationHead()
    feat = torch.randn(2, 320, 6, 6, 6)
    t_logit, n_logit, mixup_meta = head(feat)
    print(f"t_logit: {t_logit.shape}")  # (2, 4)
    print(f"n_logit: {n_logit.shape}")  # (2, 2)
    print(f"mixup_applied: {mixup_meta['mixup_applied']}")
