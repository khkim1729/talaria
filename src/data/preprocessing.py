"""
TALARIA CT Preprocessing Utilities.

Pipeline:
    1. HU Windowing : clip to [-100, 400] (liver window), normalize to [0, 1]
    2. Resampling   : resample to 1x1x1 mm isotropic via SimpleITK
    3. Patch Extract: 96x96x96 sliding window with stride 48
    4. Z-score norm : within foreground voxels

Usage:
    patches, coords, vol_shape = preprocess_ct(nifti_path, patch_size=96, stride=48)
"""

import numpy as np
import SimpleITK as sitk
from typing import List, Tuple


HU_MIN  = -100.0
HU_MAX  =  400.0
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm


def load_nifti(path: str) -> Tuple[np.ndarray, Tuple]:
    """Load NIfTI file. Returns (array [D,H,W], spacing)."""
    img     = sitk.ReadImage(path)
    spacing = img.GetSpacing()[::-1]   # SimpleITK: (W,H,D) -> numpy: (D,H,W)
    arr     = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, spacing


def resample_to_isotropic(
    arr: np.ndarray,
    spacing: Tuple,
    target_spacing: Tuple = TARGET_SPACING,
) -> np.ndarray:
    """Resample volume to isotropic voxel spacing using SimpleITK."""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(reversed(spacing)))   # numpy (D,H,W) -> sitk (W,H,D)

    orig_size    = np.array(img.GetSize(),    dtype=float)
    orig_spacing = np.array(img.GetSpacing(), dtype=float)
    new_spacing  = np.array(target_spacing[::-1], dtype=float)  # (D,H,W)->(W,H,D)
    new_size     = (orig_size * orig_spacing / new_spacing).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(HU_MIN)
    resampled = resampler.Execute(img)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def hu_window_normalize(arr: np.ndarray) -> np.ndarray:
    """Clip HU to liver window and normalize to [0, 1]."""
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    return arr.astype(np.float32)


def zscore_normalize(arr: np.ndarray, fg_threshold: float = 0.1) -> np.ndarray:
    """Z-score normalization within foreground voxels."""
    fg = arr > fg_threshold
    if fg.sum() == 0:
        return arr
    mean = arr[fg].mean()
    std  = arr[fg].std()
    if std < 1e-6:
        return arr - mean
    return ((arr - mean) / std).astype(np.float32)


def extract_patches(
    arr: np.ndarray,
    patch_size: int = 96,
    stride: int = 48,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract overlapping 3D patches via sliding window.

    Returns:
        patches: list of (patch_size, patch_size, patch_size) arrays
        coords:  list of (d, h, w) top-left corner coordinates
    """
    D, H, W = arr.shape
    P, S    = patch_size, stride
    patches, coords = [], []

    for d in range(0, max(D - P + 1, 1), S):
        for h in range(0, max(H - P + 1, 1), S):
            for w in range(0, max(W - P + 1, 1), S):
                d_end = min(d + P, D)
                h_end = min(h + P, H)
                w_end = min(w + P, W)

                patch = np.zeros((P, P, P), dtype=np.float32)
                patch[:d_end-d, :h_end-h, :w_end-w] = arr[d:d_end, h:h_end, w:w_end]
                patches.append(patch)
                coords.append((d, h, w))

    return patches, coords

# preprocessing.py에 추가할 것

def clip_and_normalize(volume, hu_min=-100.0, hu_max=400.0):
    """hu_windowing의 alias — dataset.py 호환용"""
    return hu_windowing(volume, hu_min, hu_max)

def load_mask(mask_path, target_shape=None):
    """Segmentation mask NIfTI 로드"""
    import SimpleITK as sitk
    img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(img).astype(np.uint8)
    if target_shape is not None and mask.shape != target_shape:
        from scipy.ndimage import zoom
        factors = tuple(t/s for t, s in zip(target_shape, mask.shape))
        from scipy.ndimage import zoom
        mask = zoom(mask, factors, order=0).astype(np.uint8)
    return mask


def stitch_patches(
    patch_preds: List[np.ndarray],
    coords: List[Tuple[int, int, int]],
    vol_shape: Tuple[int, int, int],
    patch_size: int = 96,
) -> np.ndarray:
    """
    Stitch patch predictions back to full volume using average overlap.

    Args:
        patch_preds: list of (patch_size, patch_size, patch_size) prediction arrays
        coords:      list of (d, h, w) top-left coordinates
        vol_shape:   (D, H, W) of the original volume
        patch_size:  P
    Returns:
        stitched: (D, H, W) averaged prediction volume
    """
    D, H, W = vol_shape
    P       = patch_size
    accum   = np.zeros((D, H, W), dtype=np.float32)
    count   = np.zeros((D, H, W), dtype=np.float32)

    for pred, (d, h, w) in zip(patch_preds, coords):
        d_end = min(d + P, D)
        h_end = min(h + P, H)
        w_end = min(w + P, W)
        accum[d:d_end, h:h_end, w:w_end] += pred[:d_end-d, :h_end-h, :w_end-w]
        count[d:d_end, h:h_end, w:w_end] += 1.0

    count = np.maximum(count, 1e-6)
    return (accum / count).astype(np.float32)


def preprocess_ct(
    nifti_path: str,
    patch_size: int = 96,
    stride: int = 48,
) -> Tuple[List[np.ndarray], List[Tuple], Tuple]:
    """
    Full preprocessing pipeline for a single CT file.

    Returns:
        patches:    list of normalized (P, P, P) float32 arrays
        coords:     list of (d, h, w) top-left coords
        vol_shape:  (D, H, W) of resampled volume
    """
    arr, spacing  = load_nifti(nifti_path)
    arr           = resample_to_isotropic(arr, spacing)
    arr           = hu_window_normalize(arr)
    arr           = zscore_normalize(arr)
    # 4D인 squeeze (e.g. (D, H, W, 1))
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr[0]
    patches, coords = extract_patches(arr, patch_size, stride)
    return patches, coords, arr.shape
