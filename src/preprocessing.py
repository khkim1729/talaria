"""
TALARIA Preprocessing Pipeline.

Steps (applied in order):
    1. HU Windowing       — clip to liver/tumor relevant HU range
    2. Isotropic Resampling — resample to 1x1x1 mm spacing
    3. Patch Extraction   — 96x96x96 overlapping patches (stride 48)
    4. Normalization      — z-score within foreground mask
"""

import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# HU Windowing
# ---------------------------------------------------------------------------

LIVER_WINDOW = (-100, 400)   # HU window for liver/tumor
SOFT_TISSUE_WINDOW = (-200, 300)


def hu_windowing(
    volume: np.ndarray,
    window_min: float = LIVER_WINDOW[0],
    window_max: float = LIVER_WINDOW[1],
) -> np.ndarray:
    """
    Clip and normalize a CT volume to [0, 1] within an HU window.

    Args:
        volume:     (D, H, W) float array in Hounsfield Units
        window_min: lower HU bound
        window_max: upper HU bound
    Returns:
        normalized: (D, H, W) float32 in [0, 1]
    """
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    return volume.astype(np.float32)


# ---------------------------------------------------------------------------
# Isotropic Resampling
# ---------------------------------------------------------------------------

def resample_to_isotropic(
    sitk_image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample an SimpleITK image to isotropic voxel spacing (default 1x1x1 mm).

    Args:
        sitk_image:      input SimpleITK image
        target_spacing:  desired voxel spacing in mm (x, y, z)
        interpolator:    SimpleITK interpolation method
    Returns:
        resampled SimpleITK image
    """
    original_spacing = sitk_image.GetSpacing()
    original_size    = sitk_image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(-1000)
    return resampler.Execute(sitk_image)


def resample_label(
    sitk_label: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """Resample a label map to match a reference image (nearest-neighbor)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(sitk_label)


# ---------------------------------------------------------------------------
# Z-score Normalization
# ---------------------------------------------------------------------------

def znorm_foreground(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Z-score normalization using foreground voxel statistics.

    Args:
        volume: (D, H, W) float32
        mask:   optional binary mask; if None, use all non-background voxels (> 0.01)
    Returns:
        normalized volume
    """
    if mask is None:
        mask = volume > 0.01
    fg = volume[mask]
    if fg.size == 0:
        return volume
    mu, sigma = fg.mean(), fg.std()
    if sigma < 1e-6:
        sigma = 1.0
    return ((volume - mu) / sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch Extraction
# ---------------------------------------------------------------------------

def extract_patches(
    volume: np.ndarray,
    patch_size: int = 96,
    stride: int = 48,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from a volume using a sliding window.

    Args:
        volume:     (D, H, W) float32
        patch_size: cubic patch size
        stride:     sliding window stride
    Returns:
        patches:    list of (patch_size, patch_size, patch_size) arrays
        coords:     list of (d, h, w) top-left corner coordinates
    """
    D, H, W = volume.shape
    P = patch_size
    patches, coords = [], []

    d_starts = list(range(0, max(D - P + 1, 1), stride))
    h_starts = list(range(0, max(H - P + 1, 1), stride))
    w_starts = list(range(0, max(W - P + 1, 1), stride))

    # Ensure the last patch covers the end
    if d_starts[-1] + P < D:
        d_starts.append(D - P)
    if h_starts[-1] + P < H:
        h_starts.append(H - P)
    if w_starts[-1] + P < W:
        w_starts.append(W - P)

    for d in d_starts:
        for h in h_starts:
            for w in w_starts:
                patch = volume[d:d+P, h:h+P, w:w+P]
                if patch.shape != (P, P, P):
                    # Pad if necessary
                    pad = [(0, max(P - patch.shape[i], 0)) for i in range(3)]
                    patch = np.pad(patch, pad, mode='constant', constant_values=0)
                patches.append(patch)
                coords.append((d, h, w))

    return patches, coords


def stitch_patches(
    patches: List[np.ndarray],
    coords: List[Tuple[int, int, int]],
    volume_shape: Tuple[int, int, int],
    patch_size: int = 96,
) -> np.ndarray:
    """
    Reconstruct a full volume from overlapping patches using average blending.

    Args:
        patches:      list of (P, P, P) arrays
        coords:       list of (d, h, w) coordinates
        volume_shape: (D, H, W) output shape
        patch_size:   P
    Returns:
        stitched: (D, H, W) float32
    """
    D, H, W = volume_shape
    P = patch_size
    accum  = np.zeros((D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    for patch, (d, h, w) in zip(patches, coords):
        d_end = min(d + P, D)
        h_end = min(h + P, H)
        w_end = min(w + P, W)
        accum[d:d_end, h:h_end, w:w_end]  += patch[:d_end-d, :h_end-h, :w_end-w]
        weight[d:d_end, h:h_end, w:w_end] += 1.0

    weight = np.maximum(weight, 1e-6)
    return (accum / weight).astype(np.float32)


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_ct(
    nifti_path: str,
    patch_size: int = 96,
    stride: int = 48,
    window_min: float = LIVER_WINDOW[0],
    window_max: float = LIVER_WINDOW[1],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]], Tuple[int, int, int]]:
    """
    Full preprocessing pipeline for a single CT NIfTI file.

    Returns:
        patches:      list of (P, P, P) float32 arrays
        coords:       patch top-left coordinates
        volume_shape: (D, H, W) of resampled volume
    """
    # Load
    sitk_img = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

    # Resample to isotropic
    sitk_img = resample_to_isotropic(sitk_img, target_spacing)

    # Convert to numpy (z, y, x -> d, h, w)
    volume = sitk.GetArrayFromImage(sitk_img)    # (D, H, W)

    # HU windowing
    volume = hu_windowing(volume, window_min, window_max)

    # Z-score normalization
    volume = znorm_foreground(volume)

    # Extract patches
    patches, coords = extract_patches(volume, patch_size, stride)

    return patches, coords, volume.shape


if __name__ == '__main__':
    # Minimal sanity check without actual files
    vol = np.random.randn(180, 200, 160).astype(np.float32) * 400
    vol_w = hu_windowing(vol)
    print(f"HU windowed: min={vol_w.min():.3f}, max={vol_w.max():.3f}")

    patches, coords = extract_patches(vol_w, patch_size=96, stride=48)
    print(f"Patches: {len(patches)}, first shape: {patches[0].shape}")

    stitched = stitch_patches(patches, coords, vol_w.shape)
    print(f"Stitched shape: {stitched.shape}")
