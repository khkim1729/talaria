"""
Phase 1: Self-Supervised Pre-training (MAE + Rotation)
Project: TALARIA (Liver Cancer TNM Staging)
Feature: Persistent Disk Caching (Stored at MSD Liver_voxel)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional, Tuple

# MONAI & Data Utilities
from monai.data import PersistentDataset, DataLoader
from monai.utils import set_determinism

# Custom Modules
from src.models.encoder import TALARIAEncoder
from src.models.decoder import ReconstructionDecoder, MaskedReconstructionModel
from src.models.rotation_head import RotationHead3D
from src.utils.rotation_3d import rotate_batch_3d
from src.data.voxel import get_liver_transforms, get_msd_liver_datalist

# 재현성을 위한 시드 고정
set_determinism(seed=42)

# ---------------------------------------------------------------------------
# Multi-Task Model Wrapper
# ---------------------------------------------------------------------------

class TALARIAPretrainModel(nn.Module):
    def __init__(self, encoder, decoder, rotation_head, mask_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.mae_model = MaskedReconstructionModel(encoder, decoder, mask_ratio)
        self.rotation_head = rotation_head

    def forward(self, x, task="mae"):
        if task == "mae":
            return self.mae_model(x)
        elif task == "rotation":
            _, deep_feat, _ = self.encoder(x)
            return self.rotation_head(deep_feat)

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def masked_recon_loss(recon, target, mask, patch_size=16):
    B, C, D, H, W = target.shape
    P = patch_size
    recon_flat = recon.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P).contiguous().view(B, -1, P**3)
    target_flat = target.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P).contiguous().view(B, -1, P**3)
    recon_flat = recon_flat[:, :mask.shape[1]]
    target_flat = target_flat[:, :mask.shape[1]]
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
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            recon, mask = model(images, task="mae")
            loss_mae = masked_recon_loss(recon, images, mask, config['token_patch_size'])
            imgs_rot, targets_rot = rotate_batch_3d(images, label_type='rand')
            rot_logits = model(imgs_rot, task="rotation")
            loss_rot = criterion_rot(rot_logits, targets_rot)
            loss = (config['w_mae'] * loss_mae) + (config['w_rot'] * loss_rot)
        optimizer.zero_grad()
        if scaler:
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
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    for batch in loader:
        images = batch['image'].to(device)
        recon, mask = model(images, task="mae")
        loss_mae = masked_recon_loss(recon, images, mask, config['token_patch_size'])
        imgs_rot, targets_rot = rotate_batch_3d(images, label_type='rand')
        rot_logits = model(imgs_rot, task="rotation")
        loss_rot = criterion_rot(rot_logits, targets_rot)
        total_mae += loss_mae.item()
        total_rot += loss_rot.item()
    random.setstate(random_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    return total_mae / len(loader), total_rot / len(loader)

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', f"pretrain_{timestamp}")
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Data Setup
    print(">>> [Data] Loading dataset list...")
    train_transforms = get_liver_transforms()
    all_files = get_msd_liver_datalist(config['data_dir'], config['json_path'])
    random.shuffle(all_files)

    split = int(len(all_files) * 0.9)
    train_files, val_files = all_files[:split], all_files[split:]

    # [수정 포인트] 캐시 경로를 datasets/MSD Liver_voxel로 설정
    # os.path.dirname을 써서 'MSD Liver'의 부모 폴더인 'datasets'로 이동한 뒤 새 폴더명을 붙여.
    parent_dir = os.path.dirname(config['data_dir'])
    cache_root = os.path.join(parent_dir, "MSD Liver_voxel")
    os.makedirs(cache_root, exist_ok=True)

    print(f">>> [Data] Persistent Cache Directory: {cache_root}")

    train_ds = PersistentDataset(
        data=train_files,
        transform=train_transforms,
        cache_dir=os.path.join(cache_root, "train")
    )
    val_ds = PersistentDataset(
        data=val_files,
        transform=train_transforms,
        cache_dir=os.path.join(cache_root, "val")
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 2. Model Setup
    model = TALARIAPretrainModel(
        TALARIAEncoder(1),
        ReconstructionDecoder(320, config['token_patch_size']),
        RotationHead3D(320, 4),
        config['mask_ratio']
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # 4. Training Loop
    best_loss = float('inf')
    print(f">>> Starting Training on {device}...")

    for epoch in range(1, config['epochs'] + 1):
        t_mae, t_rot = train_one_epoch(model, train_loader, optimizer, device, config, scaler)
        v_mae, v_rot = validate(model, val_loader, device, config)
        scheduler.step()

        val_total = (config['w_mae'] * v_mae) + (config['w_rot'] * v_rot)

        print(f"[Epoch {epoch:03d}] "
              f"Train MAE: {t_mae:.4f}, ROT: {t_rot:.4f} | "
              f"Val MAE: {v_mae:.4f}, ROT: {v_rot:.4f} | "
              f"Total: {val_total:.4f}")

        if val_total < best_loss:
            best_loss = val_total
            save_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  → Best Model Saved at {save_path}")

    print(f"\n[TALARIA] Pre-training Complete.")

if __name__ == '__main__':
    config = {
        'token_patch_size': 16,
        'batch_size': 2,
        'epochs': 100,
        'lr': 1e-4,
        'mask_ratio': 0.75,
        'w_mae': 1.0,
        'w_rot': 0.2,
        'data_dir': "/home/rintern10/talaria/datasets/MSD Liver",
        'json_path': "/home/rintern10/talaria/datasets/MSD Liver/dataset.json"
    }
    main(config)