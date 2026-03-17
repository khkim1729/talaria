"""
TALARIA Soft Voting Ensemble for Inference.

Combines TTA predictions across multiple augmentations and optionally
multiple model checkpoints via probability averaging (soft voting).

Usage:
    python -m src.inference.soft_voting \
        --config configs/finetune.yaml \
        --checkpoint experiments/finetune_<ts>/checkpoints/best.ckpt \
        --input /path/to/ct_scan.nii.gz \
        --output /path/to/output/
"""

import os
import argparse
import random
import yaml
import torch
import numpy as np
import SimpleITK as sitk
from typing import List, Optional

from src.models.talaria import TALARIANet, build_talaria
from src.data.preprocessing import preprocess_ct
from src.inference.tta import TTAPredictor, TTTAdaptor


def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single-model TTA Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: TALARIANet,
    nifti_path: str,
    patch_size: int = 96,
    stride: int = 48,
    device: torch.device = None,
    seg_threshold: float = 0.5,
    enable_ttt: bool = False,
    ttt_steps: int = 1,
    ttt_lr: float = 1e-5,
    ttt_modules: Optional[List[str]] = None,
    ttt_objective: str = 'entropy',
    ttt_reset_scope: str = 'volume',
    ttt_use_amp: bool = False,
    seed: int = 42,
) -> dict:
    """
    Full inference pipeline on a single CT NIfTI file.

    Returns:
        {
            't_seg_prob':  np.ndarray (D, H, W) tumor segmentation probability
            'n_seg_prob':  np.ndarray (D, H, W) lymph node probability
            't_seg_mask':  np.ndarray (D, H, W) binary tumor mask
            'n_seg_mask':  np.ndarray (D, H, W) binary lymph node mask
            't_stage':     str  e.g. 'T2'
            'n_stage':     str  e.g. 'N0'
            't_probs':     np.ndarray (4,) T-stage class probabilities
            'n_probs':     np.ndarray (2,) N-stage class probabilities
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_inference_seed(seed)

    patches, coords, vol_shape = preprocess_ct(nifti_path, patch_size, stride)
    patch_tensors = [torch.from_numpy(p).unsqueeze(0).unsqueeze(0) for p in patches]

    if enable_ttt:
        scope = 'patch' if ttt_reset_scope == 'patch' else 'volume'
        print(
            f"[TTT] enabled=True steps={ttt_steps} lr={ttt_lr} modules={ttt_modules} "
            f"objective={ttt_objective} reset_scope={ttt_reset_scope} amp={ttt_use_amp} device={device}"
        )
        adaptor = TTTAdaptor(
            model=model,
            steps=ttt_steps,
            lr=ttt_lr,
            adapt_modules=ttt_modules,
            objective=ttt_objective,
            reset_each_volume=True,
            use_amp=ttt_use_amp,
            device=device,
        )
        adaptor.adapt_volume(patch_tensors, scope=scope)

    predictor = TTAPredictor(model, device=device)
    print('[TTA] running final prediction with model.eval()')
    volume_preds = predictor.predict_volume(patch_tensors, coords, vol_shape, patch_size)

    t_seg_prob = volume_preds['t_seg'].squeeze().numpy()
    n_seg_prob = volume_preds['n_seg'].squeeze().numpy()
    t_cls_logits = volume_preds['t_cls'].squeeze()
    n_cls_logits = volume_preds['n_cls'].squeeze()

    t_probs = torch.softmax(t_cls_logits, dim=-1).numpy()
    n_probs = torch.softmax(n_cls_logits, dim=-1).numpy()
    t_stage = ['T1', 'T2', 'T3', 'T4'][t_probs.argmax()]
    n_stage = ['N0', 'N1'][n_probs.argmax()]

    return {
        't_seg_prob': t_seg_prob,
        'n_seg_prob': n_seg_prob,
        't_seg_mask': (t_seg_prob >= seg_threshold).astype(np.uint8),
        'n_seg_mask': (n_seg_prob >= seg_threshold).astype(np.uint8),
        't_stage':    t_stage,
        'n_stage':    n_stage,
        't_probs':    t_probs,
        'n_probs':    n_probs,
    }


# ---------------------------------------------------------------------------
# Multi-Checkpoint Soft Voting
# ---------------------------------------------------------------------------

def soft_voting_ensemble(
    config: dict,
    checkpoints: List[str],
    nifti_path: str,
    output_dir: str,
    patch_size: int = 96,
    stride: int = 48,
    seg_threshold: float = 0.5,
    enable_ttt: bool = False,
    ttt_steps: int = 1,
    ttt_lr: float = 1e-5,
    ttt_modules: Optional[List[str]] = None,
    ttt_objective: str = 'entropy',
    ttt_reset_scope: str = 'volume',
    ttt_use_amp: bool = False,
    seed: int = 42,
):
    """
    Ensemble multiple model checkpoints via soft probability averaging.

    Args:
        config:      model configuration dict
        checkpoints: list of checkpoint paths
        nifti_path:  input CT file path (.nii.gz)
        output_dir:  directory to save results
        patch_size:  patch size
        stride:      sliding window stride
        seg_threshold: binary mask threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_seg_acc = None
    n_seg_acc = None
    t_probs_acc = None
    n_probs_acc = None

    for i, ckpt_path in enumerate(checkpoints):
        print(f"[Ensemble] Running checkpoint {i+1}/{len(checkpoints)}: {ckpt_path}")
        model = build_talaria(config)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        model.eval()

        result = run_inference(
            model=model,
            nifti_path=nifti_path,
            patch_size=patch_size,
            stride=stride,
            device=device,
            seg_threshold=seg_threshold,
            enable_ttt=enable_ttt,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            ttt_modules=ttt_modules,
            ttt_objective=ttt_objective,
            ttt_reset_scope=ttt_reset_scope,
            ttt_use_amp=ttt_use_amp,
            seed=seed,
        )

        if t_seg_acc is None:
            t_seg_acc   = result['t_seg_prob'].copy()
            n_seg_acc   = result['n_seg_prob'].copy()
            t_probs_acc = result['t_probs'].copy()
            n_probs_acc = result['n_probs'].copy()
        else:
            t_seg_acc   += result['t_seg_prob']
            n_seg_acc   += result['n_seg_prob']
            t_probs_acc += result['t_probs']
            n_probs_acc += result['n_probs']

    n = len(checkpoints)
    t_seg_avg   = t_seg_acc / n
    n_seg_avg   = n_seg_acc / n
    t_probs_avg = t_probs_acc / n
    n_probs_avg = n_probs_acc / n

    t_mask = (t_seg_avg >= seg_threshold).astype(np.uint8)
    n_mask = (n_seg_avg >= seg_threshold).astype(np.uint8)
    t_stage = ['T1', 'T2', 'T3', 'T4'][t_probs_avg.argmax()]
    n_stage = ['N0', 'N1'][n_probs_avg.argmax()]

    # --- Save outputs ---
    # Reference image for spacing / origin
    ref_img = sitk.ReadImage(nifti_path)

    def save_mask(arr, fname):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(ref_img)
        sitk.WriteImage(img, os.path.join(output_dir, fname))

    def save_prob(arr, fname):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(ref_img)
        sitk.WriteImage(img, os.path.join(output_dir, fname))

    save_mask(t_mask,   'tumor_mask.nii.gz')
    save_mask(n_mask,   'lymph_mask.nii.gz')
    save_prob(t_seg_avg,'tumor_prob.nii.gz')
    save_prob(n_seg_avg,'lymph_prob.nii.gz')

    # Save TNM report
    report = {
        'T_stage': t_stage,
        'N_stage': n_stage,
        'T_probs': {f'T{i+1}': float(p) for i, p in enumerate(t_probs_avg)},
        'N_probs': {f'N{i}':   float(p) for i, p in enumerate(n_probs_avg)},
    }
    import json
    with open(os.path.join(output_dir, 'tnm_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== TALARIA Inference Report ===")
    print(f"  T-Stage: {t_stage}  (probs: {report['T_probs']})")
    print(f"  N-Stage: {n_stage}  (probs: {report['N_probs']})")
    print(f"  Outputs saved to: {output_dir}")
    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TALARIA Soft Voting Inference')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, nargs='+', required=True,
                        help='One or more checkpoint paths for ensemble')
    parser.add_argument('--input',      type=str, required=True,
                        help='Input CT NIfTI file (.nii.gz)')
    parser.add_argument('--output',     type=str, required=True,
                        help='Output directory')
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--enable_ttt', action='store_true',
                        help='Enable test-time training adaptation before TTA prediction')
    parser.add_argument('--ttt_steps', type=int, default=1,
                        help='Number of adaptation steps')
    parser.add_argument('--ttt_lr', type=float, default=1e-5,
                        help='Learning rate for TTT optimizer')
    parser.add_argument('--ttt_modules', type=str, nargs='+', default=['bn', 'head'],
                        help='Module name keywords to adapt (e.g. bn head)')
    parser.add_argument('--ttt_objective', type=str, default='entropy', choices=['entropy'],
                        help='Unsupervised TTT objective')
    parser.add_argument('--ttt_reset_scope', type=str, default='volume', choices=['volume', 'patch'],
                        help='TTT adaptation granularity')
    parser.add_argument('--ttt_use_amp', action='store_true',
                        help='Use AMP during TTT adaptation')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['mode'] = 'finetune'

    soft_voting_ensemble(
        config=config,
        checkpoints=args.checkpoint,
        nifti_path=args.input,
        output_dir=args.output,
        patch_size=config.get('patch_size', 96),
        stride=config.get('stride', 48),
        seg_threshold=args.threshold,
        enable_ttt=args.enable_ttt,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        ttt_modules=args.ttt_modules,
        ttt_objective=args.ttt_objective,
        ttt_reset_scope=args.ttt_reset_scope,
        ttt_use_amp=args.ttt_use_amp,
        seed=args.seed,
    )
