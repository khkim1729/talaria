"""
TALARIA v3 - Full Model (nnU-Net backbone, Phase 2 removed).

Architecture:
    Encoder:  TALARIAEncoder (TotalSegmentator nnU-Net pretrained → fine-tune)
    Seg Head: DualBranchSegHead (T-branch tumor seg, N-branch LN seg)
    Cls Head: ClassificationHead (T-stage 4cls, N-stage 2cls)

Phase 1 (pretrain):   TotalSegmentator weight load → fine-tune on HCC-TACE-Seg
Phase 2 (distill):    REMOVED
Phase 3 (finetune):   pretrain ckpt → dual-branch seg + classification
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.models.encoder import TALARIAEncoder
from src.models.segmentation_head import DualBranchSegHead
from src.models.classification_head import ClassificationHead


class TALARIAModel(nn.Module):
    """
    Full TALARIA model.

    Args:
        in_channels:        CT input channels (1)
        t_classes:          T-stage classes (4: T1/T2/T3/T4)
        n_classes:          N-stage classes (2: N0/N1)
        dropout:            classification head dropout
        load_totalseg:      TotalSegmentator pretrained weight 자동 로드 여부
    """

    def __init__(
        self,
        in_channels: int = 1,
        t_classes: int = 4,
        n_classes: int = 2,
        dropout: float = 0.3,
        load_totalseg: bool = False,
    ):
        super().__init__()

        self.encoder  = TALARIAEncoder(in_channels=in_channels)
        self.seg_head = DualBranchSegHead()
        self.cls_head = ClassificationHead(
            in_ch=320,
            t_classes=t_classes,
            n_classes=n_classes,
            dropout=dropout,
        )

        if load_totalseg:
            self.encoder.load_totalsegmentator_weights()

    def forward(
        self,
        x: torch.Tensor,
        apply_manifold_mixup: bool = True,
        mixup_alpha: float = 2.0,
        mixup_prob: float = 1.0,
        perm_idx: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Args:
            x: (B, 1, D, H, W)  CT volume
        Returns:
            dict with keys:
                't_seg':    (B, 1, D, H, W)   tumor segmentation logit
                'n_seg':    (B, 1, D, H, W)   LN segmentation logit
                't_cls':    (B, t_classes)     T-stage logit
                'n_cls':    (B, n_classes)     N-stage logit
                'mixup_lam': Optional[float]        manifold mixup lambda
                'mixup_perm': Optional[torch.Tensor] manifold mixup permutation
                'mixup_applied': bool               manifold mixup applied flag
        """
        shallow, deep, skips = self.encoder(x)
        t_seg, n_seg = self.seg_head(shallow, deep, skips)
        t_cls, n_cls, mixup_meta = self.cls_head(
            deep,
            apply_manifold_mixup=apply_manifold_mixup,
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
            perm_idx=perm_idx,
            lam=lam,
        )

        return {
            't_seg': t_seg,
            'n_seg': n_seg,
            't_cls': t_cls,
            'n_cls': n_cls,
            'mixup_lam': mixup_meta['mixup_lam'],
            'mixup_perm': mixup_meta['mixup_perm'],
            'mixup_applied': mixup_meta['mixup_applied'],
        }

    def load_pretrain_checkpoint(self, ckpt_path: str):
        """Phase 1 pretrain checkpoint → Phase 3 finetune용 weight 로드."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

        # encoder weight만 로드 (seg/cls head는 새로 학습)
        enc_state = {
            k.replace('encoder.', ''): v
            for k, v in state.items() if k.startswith('encoder.')
        }
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        print(f"[TALARIAModel] Pretrain ckpt loaded: {ckpt_path}")
        print(f"  encoder missing: {len(missing)}, unexpected: {len(unexpected)}")


if __name__ == '__main__':
    model = TALARIAModel(load_totalseg=False)
    vol   = torch.randn(2, 1, 96, 96, 96)
    out   = model(vol)
    for k, v in out.items():
        print(f"{k}: {v.shape if torch.is_tensor(v) else v}")
