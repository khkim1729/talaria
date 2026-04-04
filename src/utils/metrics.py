"""
TALARIA Evaluation Metrics.

수정 사항 (v5 → v5.1):
  - hausdorff95(): dead code 제거
      `pred_border = pred ^ np.zeros_like(pred)` — XOR with zeros는 pred 자체를 반환하며
      이후 pred_border가 사용되지도 않음. 삭제 후 surface distance 방법 주석 명시.
"""

import numpy as np
import torch
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Segmentation Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()
    intersection = (pred & gt).sum()
    return float(2 * intersection + smooth) / float(pred.sum() + gt.sum() + smooth)


def precision_recall(pred: np.ndarray, gt: np.ndarray,
                     smooth: float = 1e-6) -> Tuple[float, float]:
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    precision = float(tp + smooth) / float(tp + fp + smooth)
    recall    = float(tp + smooth) / float(tp + fn + smooth)
    return precision, recall


def hausdorff95(pred: np.ndarray, gt: np.ndarray,
                spacing: Tuple = (1.0, 1.0, 1.0)) -> float:
    """
    95th-percentile Hausdorff Distance (HD95) between two binary masks.

    Surface distance 계산 방법:
        1. pred/gt의 distance transform을 각각 계산
        2. pred 표면 voxel에서 gt까지의 거리 (pred_surf = dist_gt[pred])
           gt 표면 voxel에서 pred까지의 거리 (gt_surf = dist_pred[gt])
        3. 두 배열을 합쳐 95th percentile 반환
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float('nan')

    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')

    dist_pred = distance_transform_edt(~pred.astype(bool), sampling=spacing)
    dist_gt   = distance_transform_edt(~gt.astype(bool),   sampling=spacing)

    pred_surf = dist_gt[pred.astype(bool)]
    gt_surf   = dist_pred[gt.astype(bool)]

    all_surf = np.concatenate([pred_surf, gt_surf])
    return float(np.percentile(all_surf, 95))


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_auc(probs: np.ndarray, labels: np.ndarray,
                num_classes: int) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
    except ImportError:
        return float('nan')

    if len(np.unique(labels)) < 2:
        return float('nan')

    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        return float(roc_auc_score(labels_bin[:, 0], probs[:, 1]))
    return float(roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr'))


def accuracy(logits_or_probs: np.ndarray, labels: np.ndarray) -> float:
    preds = logits_or_probs.argmax(axis=-1)
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Batch Evaluation Helpers
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Running accumulator for segmentation metrics across a validation set."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._dsc  = []
        self._prec = []
        self._rec  = []
        self._hd95 = []

    def update(self, pred_prob: np.ndarray, gt: np.ndarray,
               threshold: float = 0.5, spacing: Tuple = (1.0, 1.0, 1.0)):
        pred_mask = (pred_prob >= threshold).astype(np.uint8)
        self._dsc.append(dice_score(pred_mask, gt))
        p, r = precision_recall(pred_mask, gt)
        self._prec.append(p)
        self._rec.append(r)
        self._hd95.append(hausdorff95(pred_mask, gt, spacing))

    def summary(self) -> Dict[str, float]:
        hd_vals = [v for v in self._hd95 if not np.isnan(v)]
        return {
            'DSC':       float(np.mean(self._dsc)),
            'Precision': float(np.mean(self._prec)),
            'Recall':    float(np.mean(self._rec)),
            'HD95':      float(np.mean(hd_vals)) if hd_vals else float('nan'),
        }


class ClassificationMetrics:
    """Running accumulator for classification metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._probs  = []
        self._labels = []

    def update(self, probs: np.ndarray, labels: np.ndarray):
        self._probs.append(probs)
        self._labels.append(labels)

    def summary(self) -> Dict[str, float]:
        all_probs  = np.concatenate(self._probs,  axis=0)
        all_labels = np.concatenate(self._labels, axis=0)
        return {
            'AUC':      compute_auc(all_probs, all_labels, self.num_classes),
            'Accuracy': accuracy(all_probs, all_labels),
        }


# ---------------------------------------------------------------------------
# Full Evaluation Loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, seg_threshold: float = 0.5) -> Dict:
    """TALARIANet 전체 평가."""
    model.eval()
    t_seg_metrics = SegmentationMetrics()
    n_seg_metrics = SegmentationMetrics()
    t_cls_metrics = ClassificationMetrics(num_classes=4)
    n_cls_metrics = ClassificationMetrics(num_classes=2)

    for batch in loader:
        images = batch['image'].to(device)
        out    = model(images)

        B       = images.shape[0]
        t_probs = out['t_cls'].softmax(-1).cpu().numpy()
        n_probs = out['n_cls'].softmax(-1).cpu().numpy()
        t_stage = batch.get('tstage', torch.full((B,), -1)).numpy()
        n_stage = batch.get('nstage', torch.full((B,), -1)).numpy()

        if 'label' in batch:
            gt_label = batch['label'].numpy()
            t_gt     = (gt_label == 2).astype(np.uint8)
            n_gt     = (gt_label == 3).astype(np.uint8)
            t_seg_np = out['t_seg'].sigmoid().cpu().numpy()
            n_seg_np = out['n_seg'].sigmoid().cpu().numpy()
            for b in range(B):
                t_seg_metrics.update(t_seg_np[b, 0], t_gt[b], seg_threshold)
                n_seg_metrics.update(n_seg_np[b, 0], n_gt[b], seg_threshold)

        valid = t_stage >= 0
        if valid.any():
            t_cls_metrics.update(t_probs[valid], t_stage[valid])
        valid = n_stage >= 0
        if valid.any():
            n_cls_metrics.update(n_probs[valid], n_stage[valid])

    return {
        'T_seg': t_seg_metrics.summary(),
        'N_seg': n_seg_metrics.summary(),
        'T_cls': t_cls_metrics.summary(),
        'N_cls': n_cls_metrics.summary(),
    }
