"""
Phase 1: Self-Supervised Pre-training (MAE + Rotation)
Project: TALARIA (Liver Cancer TNM Staging)

수정 사항 (v5 → v5.1):
  - argparse 추가: --config로 YAML 경로 지정 가능 (run_pretrain.sh와 연동)
  - config 키 통일: token_patch_size → pretrain.yaml 값 사용
  - masked_recon_loss: slicing 우회 제거 → shape assert로 명시적 검사
  - GradScaler: torch.amp 네임스페이스로 교체
"""

import os
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

from monai.data import PersistentDataset, DataLoader
from monai.utils import set_determinism

from src.models.encoder import TALARIAEncoder
from src.models.decoder import ReconstructionDecoder, MaskedReconstructionModel
from src.models.rotation_head import RotationHead3D
from src.utils.rotation_3d import rotate_batch_3d
from src.data.voxel import get_liver_transforms, get_msd_liver_datalist

set_determinism(seed=42)


# ---------------------------------------------------------------------------
# Multi-Task Model Wrapper
# ---------------------------------------------------------------------------

class TALARIAPretrainModel(nn.Module):
    def __init__(self, encoder, decoder, rotation_head, mask_ratio=0.5):
        super().__init__()
        self.encoder       = encoder
        self.mae_model     = MaskedReconstructionModel(encoder, decoder, mask_ratio)
        self.rotation_head = rotation_head

    def forward(self, x, task="mae"):
        if task == "mae":
            return self.mae_model(x)
        elif task == "rotation":
            _, deep_feat, _ = self.encoder(x)
            return self.rotation_head(deep_feat)
        else:
            raise ValueError(f"Unknown task: {task!r}. Use 'mae' or 'rotation'.")


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, token_patch_size: int = 16) -> torch.Tensor:
    """
    마스킹된 토큰 위치에서만 MSE loss 계산.

    Args:
        recon:            (B, 1, D, H, W) 복원 볼륨
        target:           (B, 1, D, H, W) 원본 볼륨
        mask:             (B, N) bool — True인 위치가 마스킹된 토큰
        token_patch_size: unfold 크기 (encoder 마지막 stride 배수, 기본 16)
    """
    B, C, D, H, W = target.shape
    P = token_patch_size

    recon_flat  = recon.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)\
                       .contiguous().view(B, -1, P**3)
    target_flat = target.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)\
                        .contiguous().view(B, -1, P**3)

    # shape 검사 — slicing 우회 대신 명시적 assert
    assert recon_flat.shape[1] == mask.shape[1], (
        f"patch 수 불일치: recon_flat={recon_flat.shape[1]}, mask={mask.shape[1]}. "
        f"token_patch_size={P}와 encoder 다운샘플 배수가 맞지 않습니다."
    )

    mask_bool = mask.bool().unsqueeze(-1).expand_as(recon_flat)
    return nn.functional.mse_loss(recon_flat[mask_bool], target_flat[mask_bool])


# ---------------------------------------------------------------------------
# Train / Validate Steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, config, scaler):
    model.train()
    criterion_rot = nn.CrossEntropyLoss()
    total_mae, total_rot = 0.0, 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            recon, mask = model(images, task="mae")
            loss_mae    = masked_recon_loss(recon, images, mask,
                                             config['token_patch_size'])
            imgs_rot, targets_rot = rotate_batch_3d(images, label_type='rand')
            rot_logits = model(imgs_rot, task="rotation")
            loss_rot   = criterion_rot(rot_logits, targets_rot.to(device))
            loss = config['w_mae'] * loss_mae + config['w_rot'] * loss_rot

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_mae += loss_mae.item()
        total_rot += loss_rot.item()
        pbar.set_postfix(mae=f"{loss_mae.item():.4f}", rot=f"{loss_rot.item():.4f}")

    return total_mae / len(loader), total_rot / len(loader)


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    criterion_rot = nn.CrossEntropyLoss()
    total_mae, total_rot = 0.0, 0.0

    # val 시 randomness 고정 (재현성)
    rng_state   = random.getstate()
    np_state    = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(42); torch.manual_seed(42); np.random.seed(42)

    for batch in loader:
        images = batch['image'].to(device)
        recon, mask = model(images, task="mae")
        loss_mae    = masked_recon_loss(recon, images, mask, config['token_patch_size'])
        imgs_rot, targets_rot = rotate_batch_3d(images, label_type='rand')
        rot_logits  = model(imgs_rot, task="rotation")
        loss_rot    = criterion_rot(rot_logits, targets_rot.to(device))
        total_mae  += loss_mae.item()
        total_rot  += loss_rot.item()

    # randomness 복원
    random.setstate(rng_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    return total_mae / len(loader), total_rot / len(loader)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main(config: dict):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir   = os.path.join('experiments', f"pretrain_{timestamp}")
    ckpt_dir  = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # config 스냅샷 저장
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # 1. Data Setup
    print(">>> [Data] Loading dataset list...")
    train_transforms = get_liver_transforms()
    all_files = get_msd_liver_datalist(config['data_dir'], config['json_path'])
    random.shuffle(all_files)

    split       = int(len(all_files) * 0.9)
    train_files = all_files[:split]
    val_files   = all_files[split:]

    parent_dir = os.path.dirname(config['data_dir'])
    cache_root = os.path.join(parent_dir, "MSD_Liver_voxel")
    os.makedirs(cache_root, exist_ok=True)
    print(f">>> [Data] Persistent Cache: {cache_root}")

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms,
        cache_dir=os.path.join(cache_root, "train")
    )
    val_ds = PersistentDataset(
        data=val_files, transform=train_transforms,
        cache_dir=os.path.join(cache_root, "val")
    )
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=config.get('num_workers', 8))
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=4)

    # 2. Model Setup
    token_patch_size = config['token_patch_size']
    model = TALARIAPretrainModel(
        encoder       = TALARIAEncoder(1),
        decoder       = ReconstructionDecoder(embed_dim=320, num_upsample=4),
        rotation_head = RotationHead3D(320, 4),
        mask_ratio    = config['mask_ratio'],
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config['lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # 4. Training Loop
    best_loss = float('inf')
    print(f">>> Training on {device} | epochs={config['epochs']} | "
          f"token_patch_size={token_patch_size}")

    for epoch in range(1, config['epochs'] + 1):
        t_mae, t_rot = train_one_epoch(model, train_loader, optimizer,
                                        device, config, scaler)
        v_mae, v_rot = validate(model, val_loader, device, config)
        scheduler.step()

        val_total = config['w_mae'] * v_mae + config['w_rot'] * v_rot
        print(f"[Epoch {epoch:03d}] "
              f"Train MAE={t_mae:.4f} ROT={t_rot:.4f} | "
              f"Val MAE={v_mae:.4f} ROT={v_rot:.4f} Total={val_total:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_total < best_loss:
            best_loss = val_total
            save_path = os.path.join(ckpt_dir, 'best.ckpt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         val_total,
                'config':           config,
            }, save_path)
            print(f"  ✓ Best saved: {save_path}")

        if epoch % config.get('save_every', 10) == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.ckpt'))

    print(f"\n[TALARIA] Pre-training complete. Best val loss: {best_loss:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="TALARIA Phase 1 Pre-training")
    p.add_argument('--config', type=str, default='configs/pretrain.yaml',
                   help='YAML config 경로')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"[TALARIA] Config: {args.config}")
    print(f"  token_patch_size : {config.get('token_patch_size', '(not set)')}")
    print(f"  mask_ratio       : {config.get('mask_ratio', '(not set)')}")
    main(config)