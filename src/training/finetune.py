"""
TALARIA Phase 3: Fine-tuning (nnU-Net backbone, TotalSegmentator pretrained).

Usage:
    python -m src.training.finetune \
        --config configs/finetune_96.yaml \
        [--pretrain_ckpt experiments/.../checkpoints/best.ckpt]

Config YAML keys:
    data_root:       str   HCC_TACE/ 경로
    metadata_path:   str   TNM label JSON 경로
    patch_size:      int   96 or 128
    batch_size:      int   2
    num_epochs:      int   100
    lr:              float 1e-4
    weight_decay:    float 1e-5
    experiment_dir:  str   experiments/
    load_totalseg:   bool  True  (TotalSegmentator pretrained 로드)
    t_classes:       int   4
    n_classes:       int   2
    manifold_mixup:
      enable:        bool  True
      alpha:         float 2.0
      prob:          float 1.0
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.models.talaria import TALARIAModel
from src.data.dataset import HCCTACEDataset
from src.training.losses import TALARIALoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',       required=True)
    p.add_argument('--pretrain_ckpt', default=None,
                   help='Phase 1 checkpoint (optional — overrides load_totalseg)')
    return p.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    manifold_mixup_enable=True,
    manifold_mixup_alpha=2.0,
    manifold_mixup_prob=1.0,
):
    model.train()
    total_loss = 0.0
    mixup_applied_steps = 0
    mixup_lam_sum = 0.0

    for step, batch in enumerate(loader):
        image  = batch['image'].to(device)
        tstage = batch['tstage'].to(device)
        nstage = batch['nstage'].to(device)
        seg_mask = batch.get('seg_mask')
        if seg_mask is not None:
            seg_mask = seg_mask.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(
                image,
                apply_manifold_mixup=manifold_mixup_enable,
                mixup_alpha=manifold_mixup_alpha,
                mixup_prob=manifold_mixup_prob,
            )
            mixup_lam = outputs.get('mixup_lam')
            mixup_perm = outputs.get('mixup_perm')
            loss, loss_dict = criterion(
                t_seg_logit=outputs['t_seg'],
                n_seg_logit=outputs['n_seg'],
                t_cls_logit=outputs['t_cls'],
                n_cls_logit=outputs['n_cls'],
                t_seg_gt=seg_mask,
                n_seg_gt=None,
                t_stage_gt=tstage,
                n_stage_gt=nstage,
                mixup_lam=mixup_lam,
                mixup_perm=mixup_perm,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        mixup_applied = mixup_lam is not None and mixup_perm is not None
        if mixup_applied:
            mixup_applied_steps += 1
            mixup_lam_sum += float(mixup_lam)

        loss_dict['mixup_applied'] = float(mixup_applied)
        loss_dict['lam'] = float(mixup_lam) if mixup_lam is not None else 1.0

        if step % 20 == 0:
            parts = [f"{k}={v:.4f}" for k, v in loss_dict.items()]
            print(f"  [E{epoch} S{step}] loss={loss.item():.4f} | {' '.join(parts)}")

    avg_lam = mixup_lam_sum / mixup_applied_steps if mixup_applied_steps > 0 else 1.0
    print(
        f"  [E{epoch}] mixup_applied_steps={mixup_applied_steps}/{len(loader)} "
        f"avg_lam={avg_lam:.4f}"
    )

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_t = correct_n = total = 0

    for batch in loader:
        image  = batch['image'].to(device)
        tstage = batch['tstage'].to(device)
        nstage = batch['nstage'].to(device)
        seg_mask = batch.get('seg_mask')
        if seg_mask is not None:
            seg_mask = seg_mask.to(device)

        with autocast():
            outputs = model(image)
            loss, _ = criterion(
                t_seg_logit=outputs['t_seg'],
                n_seg_logit=outputs['n_seg'],
                t_cls_logit=outputs['t_cls'],
                n_cls_logit=outputs['n_cls'],
                t_seg_gt=seg_mask,
                n_seg_gt=None,
                t_stage_gt=tstage,
                n_stage_gt=nstage,
            )

        total_loss += loss.item()
        correct_t  += (outputs['t_cls'].argmax(1) == tstage).sum().item()
        correct_n  += (outputs['n_cls'].argmax(1) == nstage).sum().item()
        total      += image.size(0)

    acc_t = correct_t / total if total > 0 else 0
    acc_n = correct_n / total if total > 0 else 0
    return total_loss / len(loader), acc_t, acc_n


def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    manifold_mixup_cfg = cfg.get('manifold_mixup', {})
    manifold_mixup_enable = manifold_mixup_cfg.get('enable', True)
    manifold_mixup_alpha  = manifold_mixup_cfg.get('alpha', 2.0)
    manifold_mixup_prob   = manifold_mixup_cfg.get('prob', 1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[finetune] device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    load_totalseg = cfg.get('load_totalseg', True) and (args.pretrain_ckpt is None)

    model = TALARIAModel(
        in_channels=cfg.get('in_channels', 1),
        t_classes=cfg.get('t_classes', 4),
        n_classes=cfg.get('n_classes', 2),
        dropout=cfg.get('dropout', 0.3),
        load_totalseg=load_totalseg,
    ).to(device)

    if args.pretrain_ckpt:
        model.load_pretrain_checkpoint(args.pretrain_ckpt)
        print(f"[finetune] Loaded pretrain ckpt: {args.pretrain_ckpt}")
    elif load_totalseg:
        print("[finetune] Using TotalSegmentator pretrained weights")
    else:
        print("[finetune] Training from scratch")

    # ── Data ─────────────────────────────────────────────────────────────────
    patch_size = cfg.get('patch_size', 96)
    train_ds = HCCTACEDataset(
        data_root=cfg['data_root'],
        metadata_path=cfg['metadata_path'],
        split='train',
        patch_size=patch_size,
        augment=True,
    )
    val_ds = HCCTACEDataset(
        data_root=cfg['data_root'],
        metadata_path=cfg['metadata_path'],
        split='val',
        patch_size=patch_size,
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 2),
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Output dirs ──────────────────────────────────────────────────────────
    exp_name = f"finetune_p{patch_size}"
    exp_dir  = os.path.join(cfg.get('experiment_dir', 'experiments'), exp_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = TALARIALoss()
    scaler    = GradScaler()

    # =========================================================
    # Phase 3a: Segmentation Head만 학습 (cls head freeze)
    # =========================================================
    seg_epochs = cfg.get('seg_epochs', 50)
    print(f"\n[Phase 3a] Segmentation-only training ({seg_epochs} epochs)")

    # cls_head freeze
    for p in model.cls_head.parameters():
        p.requires_grad = False

    optimizer_seg = torch.optim.AdamW([
        {'params': model.encoder.parameters(),  'lr': 1e-5},
        {'params': model.seg_head.parameters(), 'lr': cfg.get('seg_lr', 3e-4)},
    ], weight_decay=cfg.get('weight_decay', 1e-5))
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_seg, T_max=seg_epochs
    )

    best_seg_loss = float('inf')

    for epoch in range(1, seg_epochs + 1):
        # seg loss만 활성화 (cls weight=0)
        criterion.t_cls_w = 0.0
        criterion.n_cls_w = 0.0

        train_loss = train_one_epoch(
            model, train_loader, optimizer_seg, criterion, scaler, device, epoch,
            manifold_mixup_enable=False,   # Phase 3a에서는 mixup 끔
        )
        val_loss, acc_t, acc_n = validate(model, val_loader, criterion, device)
        scheduler_seg.step()

        print(f"[3a E{epoch:03d}/{seg_epochs}] "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"acc_T={acc_t:.3f} acc_N={acc_n:.3f} "
              f"lr={scheduler_seg.get_last_lr()[0]:.2e}")

        if val_loss < best_seg_loss:
            best_seg_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'phase': '3a',
            }, os.path.join(ckpt_dir, 'best_seg.ckpt'))
            print(f"  ✓ Best seg saved")

    # =========================================================
    # Phase 3b: Classification Head만 학습 (seg head freeze)
    # =========================================================
    cls_epochs = cfg.get('cls_epochs', 50)
    print(f"\n[Phase 3b] Classification-only training ({cls_epochs} epochs)")

    # 최선 seg checkpoint 로드
    best_seg_path = os.path.join(ckpt_dir, 'best_seg.ckpt')
    if os.path.exists(best_seg_path):
        ckpt = torch.load(best_seg_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded best seg ckpt (val_loss={ckpt['val_loss']:.4f})")

    # seg_head freeze, cls_head unfreeze
    for p in model.seg_head.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.cls_head.parameters():
        p.requires_grad = True

    optimizer_cls = torch.optim.AdamW(
        model.cls_head.parameters(),
        lr=cfg.get('cls_lr', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-5),
    )
    scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_cls, T_max=cls_epochs
    )

    # cls loss만 활성화
    criterion.t_seg_w = 0.0
    criterion.n_seg_w = 0.0
    criterion.t_cls_w = cfg.get('t_cls_weight', 0.5)
    criterion.n_cls_w = cfg.get('n_cls_weight', 0.5)

    best_val_loss = float('inf')

    for epoch in range(1, cls_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer_cls, criterion, scaler, device, epoch,
            manifold_mixup_enable=manifold_mixup_enable,
            manifold_mixup_alpha=manifold_mixup_alpha,
            manifold_mixup_prob=manifold_mixup_prob,
        )
        val_loss, acc_t, acc_n = validate(model, val_loader, criterion, device)
        scheduler_cls.step()

        print(f"[3b E{epoch:03d}/{cls_epochs}] "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"acc_T={acc_t:.3f} acc_N={acc_n:.3f} "
              f"lr={scheduler_cls.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_cls.state_dict(),
                'val_loss': val_loss,
                'acc_t': acc_t,
                'acc_n': acc_n,
                'config': cfg,
                'phase': '3b',
            }, os.path.join(ckpt_dir, 'best.ckpt'))
            print(f"  ✓ Best saved: val={val_loss:.4f} acc_T={acc_t:.3f} acc_N={acc_n:.3f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.ckpt'))

    print(f"\n[finetune] Done.")
    print(f"  Phase 3a best seg loss : {best_seg_loss:.4f}")
    print(f"  Phase 3b best val loss : {best_val_loss:.4f}")
if __name__ == '__main__':
    main()
