"""
TALARIA Reconstruction Decoder for self-supervised pretraining.
Reconstructs masked 3D patches from encoder features (Phase 1).

Architecture:
    encoder tokens (B, N, E)
        → linear projection
        → reshape to 3D feature map
        → transposed conv upsampling
        → reconstructed volume (B, 1, D, H, W)
"""

"""
TALARIA Reconstruction Decoder for self-supervised pretraining (Phase 1).
nnU-Net 인코더의 5D 피처맵을 토큰화하여 마스킹 및 복원을 수행합니다.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple


class ReconstructionDecoder(nn.Module):
    """
    토큰화된 특징(Tokens)으로부터 원본 3D CT 볼륨을 복원하는 경량 디코더.
    """

    def __init__(
        self,
        embed_dim: int = 320,     # 인코더의 최종 채널 수 (TALARIAEncoder 기준)
        patch_size: int = 16,     # 전체 볼륨 대비 최종 특징맵의 다운샘플링 배수 (96/6 = 16)
        in_channels: int = 1,     # 입력 채널 (CT = 1)
        decoder_dim: int = 128,   # 디코더 내부 채널 폭
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        # 1. 토큰을 디코더 공간으로 투영 (B, N, 320 -> B, N, 128)
        self.proj = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_dim)

        # 2. Upsampling 블록 구성
        # 16배 복원을 위해 2배 업샘플링을 4번 수행 (2^4 = 16)
        self.up_blocks = nn.ModuleList()
        ch = decoder_dim
        num_ups = int(torch.log2(torch.tensor(patch_size)).item())  # 16 -> 4
        
        for _ in range(num_ups):
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(ch, max(ch // 2, 16), kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(max(ch // 2, 16)),
                nn.GELU(),
            ))
            ch = max(ch // 2, 16)

        # 3. 최종 복원 헤드 (채널을 1로 압축)
        self.head = nn.Conv3d(ch, in_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int, int]):
        """
        Args:
            tokens: (B, N, embed_dim) - 마스킹된 토큰 뭉치
            grid:   (D', H', W') - 복원할 특징맵의 공간 해상도 (예: 6, 6, 6)
        """
        D_, H_, W_ = grid
        
        # 선형 투영 및 정규화
        x = self.proj(tokens)
        x = self.norm(x)

        # 토큰(B, N, C)을 다시 3D 특징맵(B, C, D, H, W)으로 재구성
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D_, h=H_, w=W_)

        # 단계적 업샘플링 (6x6x6 -> ... -> 96x96x96)
        for up in self.up_blocks:
            x = up(x)

        recon = self.head(x)
        return recon


class MaskedReconstructionModel(nn.Module):
    """
    Phase 1 전용 모델: 인코더 + 마스킹 + 디코더 통합.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def _mask_tokens(self, tokens: torch.Tensor):
        """임의의 토큰들을 0으로 마스킹 (MAE 방식)."""
        B, N, E = tokens.shape
        num_mask = int(N * self.mask_ratio)
        
        # 무작위 인덱스 생성
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        
        mask = torch.zeros(B, N, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        
        # 마스킹 적용
        tokens_masked = tokens.clone()
        tokens_masked[mask] = 0.0
        
        return tokens_masked, mask

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, 96, 96, 96) 원본 CT
        """
        
        _, deep_feat, _ = self.encoder(x)
        grid = deep_feat.shape[2:]  # (6, 6, 6)
        tokens = rearrange(deep_feat, 'b c d h w -> b (d h w) c')
        tokens_masked, mask = self._mask_tokens(tokens)
        recon = self.decoder(tokens_masked, grid)
        
        return recon, mask


if __name__ == '__main__':
    # 테스트 환경 구축
    from src.models.encoder import TALARIAEncoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = TALARIAEncoder(in_channels=1).to(device)
    dec = ReconstructionDecoder(embed_dim=320, patch_size=16).to(device)
    model = MaskedReconstructionModel(enc, dec, mask_ratio=0.75).to(device)
   
    vol = torch.randn(2, 1, 96, 96, 96).to(device) 
    print(f"입력 데이터 모양: {vol.shape}")
    recon, mask = model(vol)
    print(f"recon: {recon.shape}, mask: {mask.shape}")