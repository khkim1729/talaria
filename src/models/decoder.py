"""
TALARIA Reconstruction Decoder for self-supervised pretraining (Phase 1).
nnU-Net 인코더의 5D 피처맵을 토큰화하여 마스킹 및 복원을 수행합니다.

수정 사항 (v5 → v5.1):
  - patch_size 기반 log2() 계산 제거 → num_upsample=4 명시 고정
    TALARIAEncoder는 stride=2 × 4단계 = 16배 다운샘플이므로
    96 입력 → 6×6×6 feature → 2^4=16배 업샘플로 96 복원이 보장됨.
    log2(patch_size) 방식은 patch_size=96, 128 등 잘못 전달 시 깨짐.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple


# TALARIAEncoder 다운샘플 배수 (stride=2 × 4단계)
_ENCODER_DOWNSAMPLE = 16   # 96 → 6, 128 → 8
_NUM_UPSAMPLE       = 4    # 2^4 = 16배 복원


class ReconstructionDecoder(nn.Module):
    """
    토큰화된 특징(Tokens)으로부터 원본 3D CT 볼륨을 복원하는 경량 디코더.

    Args:
        embed_dim:   인코더 최종 채널 수 (TALARIAEncoder 고정: 320)
        num_upsample: 업샘플링 횟수. 기본값 4 (2^4=16배, encoder downscale 역산)
        in_channels: 입력 채널 (CT = 1)
        decoder_dim: 디코더 내부 채널 폭
    """

    def __init__(
        self,
        embed_dim: int = 320,
        num_upsample: int = _NUM_UPSAMPLE,   # ← patch_size 대신 명시적 횟수
        in_channels: int = 1,
        decoder_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels

        # 1. 토큰을 디코더 공간으로 투영 (B, N, embed_dim -> B, N, decoder_dim)
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_dim)

        # 2. Upsampling 블록 (num_upsample번, 각 2배)
        self.up_blocks = nn.ModuleList()
        ch = decoder_dim
        for _ in range(num_upsample):
            out_ch = max(ch // 2, 16)
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch),
                nn.GELU(),
            ))
            ch = out_ch

        # 3. 최종 복원 헤드
        self.head = nn.Conv3d(ch, in_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, embed_dim) — 마스킹된 토큰
            grid:   (D', H', W')      — 피처맵 공간 해상도 (예: 6, 6, 6)
        Returns:
            recon: (B, in_channels, D, H, W)
        """
        D_, H_, W_ = grid

        x = self.proj(tokens)
        x = self.norm(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D_, h=H_, w=W_)

        for up in self.up_blocks:
            x = up(x)

        return self.head(x)


class MaskedReconstructionModel(nn.Module):
    """Phase 1 전용 모델: 인코더 + 마스킹 + 디코더 통합."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.mask_ratio = mask_ratio

    def _mask_tokens(self, tokens: torch.Tensor):
        """임의의 토큰들을 0으로 마스킹 (MAE 방식)."""
        B, N, E = tokens.shape
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)

        mask = torch.zeros(B, N, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)

        tokens_masked = tokens.clone()
        tokens_masked[mask] = 0.0
        return tokens_masked, mask

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, D, H, W) 원본 CT
        Returns:
            recon: (B, 1, D, H, W) 복원된 볼륨
            mask:  (B, N)          마스킹된 토큰 위치
        """
        _, deep_feat, _ = self.encoder(x)
        grid   = deep_feat.shape[2:]                             # (6, 6, 6) for 96^3 input
        tokens = rearrange(deep_feat, 'b c d h w -> b (d h w) c')

        # shape 검사 — 업샘플 후 입력 크기와 일치하는지 확인
        expected_out = tuple(g * (2 ** len(self.decoder.up_blocks)) for g in grid)
        assert expected_out == x.shape[2:], (
            f"Decoder output shape {expected_out} != input shape {x.shape[2:]}. "
            f"num_upsample={len(self.decoder.up_blocks)}이 encoder downscale과 일치하지 않습니다."
        )

        tokens_masked, mask = self._mask_tokens(tokens)
        recon = self.decoder(tokens_masked, grid)
        return recon, mask


if __name__ == '__main__':
    from src.models.encoder import TALARIAEncoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc   = TALARIAEncoder(in_channels=1).to(device)
    dec   = ReconstructionDecoder(embed_dim=320, num_upsample=4).to(device)
    model = MaskedReconstructionModel(enc, dec, mask_ratio=0.75).to(device)

    vol = torch.randn(2, 1, 96, 96, 96).to(device)
    recon, mask = model(vol)
    print(f"recon: {recon.shape}")   # (2, 1, 96, 96, 96)
    print(f"mask:  {mask.shape}")    # (2, 216)
