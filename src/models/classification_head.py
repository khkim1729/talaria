"""
TALARIA Classification Head.

수정 사항 (v5 → v5.1):
  - MorphologicalFeatureExtractor._connected_components_3d():
      Python 3중 리스트 BFS → scipy.ndimage.label 으로 교체.
      96^3 voxel에서 C extension 속도로 connected component 계산.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Morphological Feature Extractor (T-Stage)
# ---------------------------------------------------------------------------

class MorphologicalFeatureExtractor(nn.Module):
    """
    Tumor segmentation mask에서 T-stage 분류에 필요한 형태적 특징 추출.

    AJCC 8th edition T-stage criteria 기반:
        - maximum lesion diameter (cm)
        - lesion count
        - vascular invasion proxy  (간문맥/간정맥 근접도)
        - hepatic lobe involvement fraction

    Output: (B, 4) float32
    """

    def __init__(self, voxel_spacing_mm: float = 1.0):
        super().__init__()
        self.voxel_spacing_mm = voxel_spacing_mm
        self.max_diam_norm    = 100.0   # 10cm = AJCC T4 상한 기준
        self.max_count_norm   = 5.0

    @torch.no_grad()
    def _connected_components_3d(self, binary: torch.Tensor) -> Tuple[int, float]:
        """
        scipy.ndimage.label 기반 connected component 분석.
        binary: (D, H, W) bool tensor

        Returns:
            n_components: int
            max_diam_vox: float (최대 bounding box 대각선 길이, voxel 단위)
        """
        try:
            from scipy.ndimage import label as scipy_label
            import numpy as np

            mask_np = binary.cpu().numpy().astype(bool)
            labeled, n_comp = scipy_label(mask_np)

            if n_comp == 0:
                return 0, 0.0

            max_diam = 0.0
            for c in range(1, n_comp + 1):
                vox = np.argwhere(labeled == c)
                if len(vox) == 0:
                    continue
                dmin, dmax = vox[:, 0].min(), vox[:, 0].max()
                hmin, hmax = vox[:, 1].min(), vox[:, 1].max()
                wmin, wmax = vox[:, 2].min(), vox[:, 2].max()
                diam = float(((dmax-dmin)**2 + (hmax-hmin)**2 + (wmax-wmin)**2) ** 0.5)
                max_diam = max(max_diam, diam)

            return n_comp, max_diam

        except ImportError:
            # scipy 없는 환경 fallback — 단순 전체 마스크 bounding box
            mask_np = binary.cpu().numpy().astype(bool)
            if not mask_np.any():
                return 0, 0.0
            import numpy as np
            vox = np.argwhere(mask_np)
            diam = float(((vox[:, 0].ptp())**2 + (vox[:, 1].ptp())**2 + (vox[:, 2].ptp())**2) ** 0.5)
            return 1, diam

    def _soft_features(self, prob: torch.Tensor) -> torch.Tensor:
        """
        학습 중 미분 가능한 soft feature 추출.
        prob: (D, H, W) float [0,1]
        Returns: (4,) tensor
        """
        D, H, W = prob.shape

        total = prob.sum().clamp(min=1e-6)
        d_idx = torch.arange(D, device=prob.device, dtype=prob.dtype)
        h_idx = torch.arange(H, device=prob.device, dtype=prob.dtype)
        w_idx = torch.arange(W, device=prob.device, dtype=prob.dtype)

        mean_d = (prob.sum(dim=(1,2)) * d_idx).sum() / total
        mean_h = (prob.sum(dim=(0,2)) * h_idx).sum() / total
        mean_w = (prob.sum(dim=(0,1)) * w_idx).sum() / total

        var_d = ((d_idx - mean_d)**2 * prob.sum(dim=(1,2))).sum() / total
        var_h = ((h_idx - mean_h)**2 * prob.sum(dim=(0,2))).sum() / total
        var_w = ((w_idx - mean_w)**2 * prob.sum(dim=(0,1))).sum() / total

        diam_vox  = 2.0 * (torch.stack([var_d, var_h, var_w]).max().clamp(min=0.0).sqrt())
        diam_mm   = diam_vox * self.voxel_spacing_mm
        diam_norm = (diam_mm / self.max_diam_norm).clamp(0.0, 1.0)

        volume      = prob.sum()
        count_proxy = (volume / (4.0/3.0 * 3.14159 * (10.0**3))).clamp(0.0, self.max_count_norm)
        count_norm  = count_proxy / self.max_count_norm

        d1, d2 = D//3, 2*D//3
        h1, h2 = H//3, 2*H//3
        w1, w2 = W//3, 2*W//3
        vasc_proxy = (prob[d1:d2, h1:h2, w1:w2].sum() / total).clamp(0.0, 1.0)

        left_mass  = prob[:, :, :W//2].sum()
        right_mass = prob[:, :, W//2:].sum()
        lobe_frac  = (torch.min(left_mass, right_mass) /
                      (torch.max(left_mass, right_mass).clamp(min=1e-6))).clamp(0.0, 1.0)

        return torch.stack([diam_norm, count_norm, vasc_proxy, lobe_frac])

    def forward(self, seg_prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            seg_prob: (B, 1, D, H, W) T-Branch sigmoid output
        Returns:
            morph_feat: (B, 4) float32
        """
        B = seg_prob.shape[0]
        feats = []

        for b in range(B):
            prob = seg_prob[b, 0]   # (D, H, W)

            if self.training:
                feat = self._soft_features(prob)
            else:
                binary = (prob > threshold)
                n_comp, max_diam_vox = self._connected_components_3d(binary)

                diam_mm    = max_diam_vox * self.voxel_spacing_mm
                diam_norm  = min(diam_mm / self.max_diam_norm, 1.0)
                count_norm = min(n_comp / self.max_count_norm, 1.0)

                total = prob.sum().clamp(min=1e-6)
                D, H, W = prob.shape
                d1, d2 = D//3, 2*D//3
                h1, h2 = H//3, 2*H//3
                w1, w2 = W//3, 2*W//3
                vasc_proxy = (prob[d1:d2, h1:h2, w1:w2].sum() / total).clamp(0.0, 1.0)

                left  = prob[:, :, :W//2].sum()
                right = prob[:, :, W//2:].sum()
                lobe_frac = (torch.min(left, right) /
                             torch.max(left, right).clamp(min=1e-6)).clamp(0.0, 1.0)

                feat = torch.tensor(
                    [diam_norm, count_norm, float(vasc_proxy), float(lobe_frac)],
                    dtype=seg_prob.dtype, device=seg_prob.device
                )

            feats.append(feat)

        return torch.stack(feats, dim=0)   # (B, 4)


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Dual classification head.

    T-Stage: MorphologicalFeatureExtractor(seg_prob) -> 3-layer MLP -> T1~T4
    N-Stage: GAP(deep_feat) -> MLP -> N0/N1  (Manifold Mixup 지원)
    """

    def __init__(
        self,
        in_ch: int = 320,
        t_classes: int = 4,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.morph_extractor = MorphologicalFeatureExtractor()
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.t_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, t_classes),
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
        t_seg_prob: torch.Tensor,
        apply_manifold_mixup: bool = True,
        mixup_alpha: float = 2.0,
        mixup_prob: float = 1.0,
        perm_idx: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ):
        morph_feat = self.morph_extractor(t_seg_prob)
        t_logit    = self.t_head(morph_feat)

        x = self.gap(deep_feat).flatten(1)   # (B, 320)

        mixup_applied = False
        mixup_lam: Optional[float] = None
        mixup_perm: Optional[torch.Tensor] = None

        bsz      = x.size(0)
        can_mixup = bsz > 1 and mixup_prob > 0.0 and mixup_alpha > 0.0

        if (self.training and apply_manifold_mixup and can_mixup
                and torch.rand((), device=x.device) < min(float(mixup_prob), 1.0)):
            mixup_perm = perm_idx if perm_idx is not None \
                else torch.randperm(bsz, device=x.device)
            if lam is None:
                mixup_lam = float(torch.distributions.Beta(
                    mixup_alpha, mixup_alpha).sample().item())
            else:
                mixup_lam = float(lam)
            x = mixup_lam * x + (1.0 - mixup_lam) * x[mixup_perm]
            mixup_applied = True

        n_logit = self.n_head(x)

        mixup_meta: Dict[str, Any] = {
            'mixup_lam':     mixup_lam,
            'mixup_perm':    mixup_perm,
            'mixup_applied': mixup_applied,
        }
        return t_logit, n_logit, mixup_meta


if __name__ == '__main__':
    head       = ClassificationHead()
    deep_feat  = torch.randn(2, 320, 6, 6, 6)
    t_seg_prob = torch.sigmoid(torch.randn(2, 1, 96, 96, 96))

    head.eval()
    t_logit, n_logit, meta = head(deep_feat, t_seg_prob)
    print(f"t_logit: {t_logit.shape}")
    print(f"n_logit: {n_logit.shape}")
    print(f"mixup:   {meta}")
