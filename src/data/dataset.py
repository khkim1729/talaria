"""
TALARIA Dataset Loaders.

지원 데이터셋:
    - LiTS / MSD Liver    : Phase 1 pretraining + T-Branch seg 학습
    - TCGA-LIHC           : Phase 1 pretraining
    - HCC-TACE-Seg        : Phase 3 finetune (seg mask + TNM label)
    - Mediastinal-LN-Seg  : N-Branch pretraining (림프절 annotation)

공통 출력 형식:
    pretrain  → {'image': (1,D,H,W) float32}
    finetune  → {'image': (1,D,H,W), 'tstage': int, 'nstage': int,
                  'seg_mask': (1,D,H,W) float32 or None}
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    preprocess_ct, load_mask, clip_and_normalize, extract_patches
)

try:
    from scipy.ndimage import map_coordinates, gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import nibabel as nib
except ImportError:
    nib = None


# ---------------------------------------------------------------------------
# Augmentation (학습 전용)
# ---------------------------------------------------------------------------

def random_augment(volume: np.ndarray) -> np.ndarray:
    """
    간단한 3D 증강 (이미 [0,1] 정규화된 volume에 적용).
        - random flip (각 축 50%)
        - intensity jitter (±10%)
        - Gaussian noise (std=0.02)
    """
    for axis in [0, 1, 2]:
        if random.random() < 0.5:
            volume = np.flip(volume, axis=axis).copy()

    scale = random.uniform(0.9, 1.1)
    shift = random.uniform(-0.05, 0.05)
    volume = np.clip(volume * scale + shift, 0.0, 1.0)

    volume = volume + np.random.randn(*volume.shape).astype(np.float32) * 0.02
    volume = np.clip(volume, 0.0, 1.0)

    return volume.astype(np.float32)


def elastic_deformation_3d(
    volume: np.ndarray,
    alpha: float = 8.0,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    3D Elastic deformation (Simard et al. 2003).
    림프절처럼 작고 부드러운 구조의 shape variability를 시뮬레이션.

    Args:
        volume: (D, H, W) float32 [0,1]
        alpha:  deformation 강도 (클수록 강함, 림프절용 8~15 권장)
        sigma:  Gaussian smoothing sigma (작을수록 국소적 변형)
    Returns:
        deformed (D, H, W) float32
    """
    if not SCIPY_AVAILABLE:
        return volume

    D, H, W = volume.shape

    # random displacement field
    dx = gaussian_filter(
        np.random.randn(D, H, W).astype(np.float32) * alpha, sigma)
    dy = gaussian_filter(
        np.random.randn(D, H, W).astype(np.float32) * alpha, sigma)
    dz = gaussian_filter(
        np.random.randn(D, H, W).astype(np.float32) * alpha, sigma)

    z, y, x = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij')

    coords = [
        np.clip(z + dx, 0, D - 1).ravel(),
        np.clip(y + dy, 0, H - 1).ravel(),
        np.clip(x + dz, 0, W - 1).ravel(),
    ]

    deformed = map_coordinates(volume, coords, order=1, mode='nearest')
    return deformed.reshape(D, H, W).astype(np.float32)


def n1_heavy_augment(volume: np.ndarray) -> np.ndarray:
    """
    N1 케이스 전용 heavy augmentation.
    N1 데이터 부족(14/105) 문제를 극복하기 위해
    더 aggressive한 변환을 적용하여 데이터 다양성 확보.

    적용 순서:
        1. Elastic deformation (항상)
        2. Random flip (각 축 70%)
        3. Random 90도 rotation
        4. Intensity jitter (±15%, 더 강함)
        5. Gaussian noise (std=0.03)
        6. Gamma correction
    """
    # 1. Elastic deformation (항상 적용)
    alpha = random.uniform(8.0, 15.0)
    sigma = random.uniform(2.5, 4.0)
    volume = elastic_deformation_3d(volume, alpha=alpha, sigma=sigma)

    # 2. Random flip (70% — 기본보다 높음)
    for axis in [0, 1, 2]:
        if random.random() < 0.7:
            volume = np.flip(volume, axis=axis).copy()

    # 3. Random 90도 rotation
    if random.random() < 0.5:
        k = random.randint(1, 3)
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        volume = np.rot90(volume, k=k, axes=axes).copy()

    # 4. Intensity jitter (±15%)
    scale = random.uniform(0.85, 1.15)
    shift = random.uniform(-0.08, 0.08)
    volume = np.clip(volume * scale + shift, 0.0, 1.0)

    # 5. Gaussian noise
    volume = volume + np.random.randn(*volume.shape).astype(np.float32) * 0.03
    volume = np.clip(volume, 0.0, 1.0)

    # 6. Gamma correction (CT contrast 변화 시뮬레이션)
    gamma = random.uniform(0.7, 1.4)
    volume = np.power(volume, gamma).astype(np.float32)

    return volume


# ---------------------------------------------------------------------------
# LiTS / MSD Liver Dataset (Phase 1 pretraining + T-Branch)
# ---------------------------------------------------------------------------

class LiTSDataset(Dataset):
    """
    LiTS (Medical Segmentation Decathlon Task03_Liver) 데이터셋.

    디렉토리 구조:
        lits_root/
            imagesTr/  ← CT NIfTI (.nii.gz)
            labelsTr/  ← seg mask NIfTI (0=bg, 1=liver, 2=tumor)

    Args:
        root:       Task03_Liver 루트 경로
        split:      'train' | 'val' | 'all'
        patch_size: 96 or 128
        stride:     sliding stride
        augment:    학습용 증강 여부
        mode:       'pretrain' (image only) | 'seg' (image + mask)
    """

    TRAIN_RATIO = 0.85

    def __init__(
        self,
        root: str,
        split: str = 'train',
        patch_size: int = 96,
        stride: int = 48,
        augment: bool = False,
        mode: str = 'pretrain',
    ):
        self.root       = Path(root)
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment
        self.mode       = mode

        img_dir  = self.root / 'imagesTr'
        lbl_dir  = self.root / 'labelsTr'

        all_imgs = sorted(img_dir.glob('*.nii.gz'))
        n_train  = int(len(all_imgs) * self.TRAIN_RATIO)

        if split == 'train':
            self.img_paths = all_imgs[:n_train]
        elif split == 'val':
            self.img_paths = all_imgs[n_train:]
        else:
            self.img_paths = all_imgs

        self.lbl_paths = [
            lbl_dir / p.name for p in self.img_paths
        ] if mode == 'seg' else [None] * len(self.img_paths)

        # patch index 미리 계산 (lazy loading)
        self._patch_index: List[Tuple[int, int]] = []   # (img_idx, patch_idx)
        self._patch_cache: Dict[int, Tuple] = {}         # img_idx → (patches, masks)
        self._build_index()

    def _build_index(self):
        """각 volume의 예상 patch 수를 계산해서 index 구성."""
        # 실제 로드 없이 추정 (첫 번째 volume으로 calibrate)
        # 실제로는 lazy하게 __getitem__에서 cache
        # 일단 volume당 하나씩 index (첫 접근 시 expand)
        for i in range(len(self.img_paths)):
            self._patch_index.append((i, 0))

    def _load_volume(self, img_idx: int):
        if img_idx in self._patch_cache:
            return self._patch_cache[img_idx]

        img_path = str(self.img_paths[img_idx])
        patches, coords, shape = preprocess_ct(
            img_path, self.patch_size, self.stride
        )

        masks = None
        if self.mode == 'seg' and self.lbl_paths[img_idx] is not None:
            mask_vol = load_mask(str(self.lbl_paths[img_idx]), shape)
            # tumor label만 추출 (0=bg, 1=liver, 2=tumor → tumor=1)
            tumor_mask = (mask_vol == 2).astype(np.float32)
            mask_patches, _ = extract_patches(tumor_mask, self.patch_size, self.stride)
            masks = mask_patches

        result = (patches, masks, coords, shape)
        self._patch_cache[img_idx] = result

        # index 재구성
        for j in range(len(patches)):
            if (img_idx, j) not in self._patch_index:
                self._patch_index.append((img_idx, j))

        return result

    def __len__(self):
        return len(self._patch_index)

    def __getitem__(self, idx: int) -> dict:
        img_idx, patch_idx = self._patch_index[idx]

        # 첫 접근 시 volume 로드
        if img_idx not in self._patch_cache:
            # 단순하게: volume 전체 patch 중 random하게 하나
            patches, masks, coords, shape = self._load_volume(img_idx)
            patch_idx = random.randint(0, len(patches) - 1)
        else:
            patches, masks, coords, shape = self._patch_cache[img_idx]
            patch_idx = min(patch_idx, len(patches) - 1)

        image = patches[patch_idx].copy()   # (1, P, P, P)

        if self.augment:
            image[0] = random_augment(image[0])

        out = {'image': torch.from_numpy(image)}

        if self.mode == 'seg' and masks is not None:
            seg = masks[patch_idx].copy()   # (1, P, P, P)
            out['seg_mask'] = torch.from_numpy(seg)

        return out


# ---------------------------------------------------------------------------
# HCC-TACE-Seg Dataset (Phase 3 finetune)
# ---------------------------------------------------------------------------

class HCCTACEDataset(Dataset):
    """
    HCC-TACE-Seg 데이터셋 (Phase 3 fine-tuning).

    디렉토리 구조 (TCIA 다운로드 후 변환):
        data_root/
            HCC_001/
                image.nii.gz       ← pre-TACE CT (arterial phase)
                seg_mask.nii.gz    ← tumor segmentation mask
            HCC_002/
                ...

    TNM label JSON 형식:
        {
            "HCC_001": {"T": 2, "N": 0, "M": 0},
            "HCC_002": {"T": 1, "N": 0, "M": 0},
            ...
        }
        T: 1~4 (int), N: 0~1 (int)

    Args:
        data_root:     HCC-TACE 루트 경로
        metadata_path: TNM label JSON 경로
        split:         'train' | 'val'
        patch_size:    96 or 128
        augment:       학습용 증강
    """

    TRAIN_RATIO = 0.80

    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        split: str = 'train',
        patch_size: int = 96,
        stride: int = 48,
        augment: bool = False,
        n1_oversample_ratio: float = 5.0,  # N1 케이스를 N0 대비 몇 배 더 샘플링할지
    ):
        self.data_root  = Path(data_root)
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment
        self.n1_oversample_ratio = n1_oversample_ratio

        with open(metadata_path) as f:
            self.labels: Dict = json.load(f)

        all_cases = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and d.name in self.labels
            and (d / 'image.nii.gz').exists()
        ])

        n_train = int(len(all_cases) * self.TRAIN_RATIO)
        random.seed(42)
        random.shuffle(all_cases)

        if split == 'train':
            self.cases = all_cases[:n_train]
        else:
            self.cases = all_cases[n_train:]
            self.n1_oversample_ratio = 1.0  # val은 oversampling 안 함

        # N0/N1 케이스 분리
        self.n0_indices = [
            i for i, c in enumerate(self.cases)
            if self.labels[c.name]['N'] == 0
        ]
        self.n1_indices = [
            i for i, c in enumerate(self.cases)
            if self.labels[c.name]['N'] == 1
        ]

        # Oversampling index 구성
        # N1을 ratio배 반복 → 학습 중 N1 노출 빈도 증가
        if split == 'train' and self.n1_indices:
            n1_repeat = int(len(self.n1_indices) * n1_oversample_ratio)
            self._sample_indices = (
                self.n0_indices +
                [self.n1_indices[i % len(self.n1_indices)]
                 for i in range(n1_repeat)]
            )
            random.shuffle(self._sample_indices)
        else:
            self._sample_indices = list(range(len(self.cases)))

        n1_count = len(self.n1_indices)
        n0_count = len(self.n0_indices)
        print(f"[HCCTACEDataset] {split}: N0={n0_count}, N1={n1_count}, "
              f"effective_len={len(self._sample_indices)}")

        self._cache: Dict[int, Tuple] = {}

    def _load_case(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        case_dir = self.cases[idx]
        img_path = str(case_dir / 'image.nii.gz')
        patches, coords, shape = preprocess_ct(
            img_path, self.patch_size, self.stride
        )

        seg_patches = None
        seg_path = case_dir / 'seg_mask.nii.gz'
        if seg_path.exists():
            mask_vol = load_mask(str(seg_path), shape)
            tumor_mask = (mask_vol > 0).astype(np.float32)
            seg_patches, _ = extract_patches(tumor_mask, self.patch_size, self.stride)

        label = self.labels[case_dir.name]
        t_stage = int(label['T']) - 1   # 0-indexed (T1→0, T2→1, T3→2, T4→3)
        n_stage = int(label['N'])        # 0 or 1

        result = (patches, seg_patches, t_stage, n_stage)
        self._cache[idx] = result
        return result

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> dict:
        case_idx = self._sample_indices[idx]
        patches, seg_patches, t_stage, n_stage = self._load_case(case_idx)

        if self.augment:
            p_idx = random.randint(0, len(patches) - 1)
        else:
            p_idx = len(patches) // 2

        image = patches[p_idx].copy()   # (1, P, P, P)

        if self.augment:
            if n_stage == 1:
                # N1 케이스: heavy augmentation (elastic deformation 포함)
                image[0] = n1_heavy_augment(image[0])
            else:
                # N0 케이스: 기본 augmentation
                image[0] = random_augment(image[0])

        out = {
            'image':  torch.from_numpy(image),
            'tstage': torch.tensor(t_stage, dtype=torch.long),
            'nstage': torch.tensor(n_stage, dtype=torch.long),
        }

        if seg_patches is not None:
            out['seg_mask'] = torch.from_numpy(seg_patches[p_idx].copy())

        return out


# ---------------------------------------------------------------------------
# Mediastinal Lymph Node Dataset (N-Branch pretraining)
# ---------------------------------------------------------------------------

class MediastinalLNDataset(Dataset):
    """
    Mediastinal-Lymph-Node-Seg 데이터셋.
    N-Branch pretraining용 림프절 segmentation 데이터.

    디렉토리 구조 (TCIA 다운로드 후 변환):
        data_root/
            patient_001/
                image.nii.gz
                lymph_mask.nii.gz   ← 림프절 segmentation
            ...

    Args:
        data_root:  데이터 루트
        split:      'train' | 'val'
        patch_size: 96 or 128
        augment:    증강 여부
    """

    TRAIN_RATIO = 0.85

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 96,
        stride: int = 48,
        augment: bool = False,
    ):
        self.data_root  = Path(data_root)
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment

        all_cases = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d / 'image.nii.gz').exists()
        ])

        n_train = int(len(all_cases) * self.TRAIN_RATIO)
        if split == 'train':
            self.cases = all_cases[:n_train]
        else:
            self.cases = all_cases[n_train:]

        self._cache: Dict[int, Tuple] = {}

    def _load_case(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        case_dir = self.cases[idx]
        patches, coords, shape = preprocess_ct(
            str(case_dir / 'image.nii.gz'), self.patch_size, self.stride
        )

        mask_patches = None
        mask_path = case_dir / 'lymph_mask.nii.gz'
        if mask_path.exists():
            mask_vol = load_mask(str(mask_path), shape)
            ln_mask  = (mask_vol > 0).astype(np.float32)
            mask_patches, _ = extract_patches(ln_mask, self.patch_size, self.stride)

        self._cache[idx] = (patches, mask_patches)
        return patches, mask_patches

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        patches, mask_patches = self._load_case(idx)
        p_idx = random.randint(0, len(patches) - 1) if self.augment \
                else len(patches) // 2

        image = patches[p_idx].copy()
        if self.augment:
            image[0] = random_augment(image[0])

        out = {'image': torch.from_numpy(image)}
        if mask_patches is not None:
            out['seg_mask'] = torch.from_numpy(mask_patches[p_idx].copy())
        return out


# ---------------------------------------------------------------------------
# Combined Pretrain Dataset (LiTS + TCGA-LIHC)
# ---------------------------------------------------------------------------

class CombinedPretrainDataset(Dataset):
    """
    Phase 1 pretraining용 통합 데이터셋.
    LiTS + TCGA-LIHC를 합쳐서 하나의 Dataset으로 제공.

    AMOS는 optional (없으면 skip).
    """

    def __init__(self, datasets: list):
        self.datasets   = datasets
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx: int) -> dict:
        for i, cum in enumerate(self.cumulative):
            if idx < cum:
                offset = idx - (self.cumulative[i-1] if i > 0 else 0)
                return self.datasets[i][offset]
        raise IndexError(idx)


# ---------------------------------------------------------------------------
# TCGA-LIHC Dataset (pretrain용, label 없이 image만)
# ---------------------------------------------------------------------------

class TCGALIHCDataset(Dataset):
    """
    TCGA-LIHC pretraining 데이터셋 (image only).

    디렉토리 구조:
        data_root/
            TCGA-XX-XXXX/
                image.nii.gz
            ...
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int = 96,
        stride: int = 48,
        augment: bool = False,
    ):
        self.data_root  = Path(data_root)
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment

        self.cases = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and (d / 'image.nii.gz').exists()
        ])
        self._cache: Dict[int, List] = {}

    def _load_case(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        patches, _, _ = preprocess_ct(
            str(self.cases[idx] / 'image.nii.gz'),
            self.patch_size, self.stride
        )
        self._cache[idx] = patches
        return patches

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        patches = self._load_case(idx)
        p_idx   = random.randint(0, len(patches) - 1)
        image   = patches[p_idx].copy()
        if self.augment:
            image[0] = random_augment(image[0])
        return {'image': torch.from_numpy(image)}


# ---------------------------------------------------------------------------
# build_pretrain_dataset (pretrain.py에서 호출)
# ---------------------------------------------------------------------------

def build_pretrain_dataset(
    lits_root:  Optional[str] = None,
    tcia_root:  Optional[str] = None,
    amos_root:  Optional[str] = None,   # 현재 미사용, 추후 추가
    patch_size: int = 96,
    stride:     int = 48,
) -> Dataset:
    """
    Phase 1 pretraining용 통합 데이터셋 빌드.
    사용 가능한 데이터만 포함 (None이면 skip).
    """
    datasets = []

    if lits_root and os.path.isdir(lits_root):
        datasets.append(LiTSDataset(
            lits_root, split='all', patch_size=patch_size,
            stride=stride, augment=True, mode='pretrain'
        ))
        print(f"[dataset] LiTS loaded: {lits_root}")
    else:
        print(f"[dataset] LiTS skipped (not found: {lits_root})")

    if tcia_root and os.path.isdir(tcia_root):
        datasets.append(TCGALIHCDataset(
            tcia_root, patch_size=patch_size, stride=stride, augment=True
        ))
        print(f"[dataset] TCGA-LIHC loaded: {tcia_root}")
    else:
        print(f"[dataset] TCGA-LIHC skipped (not found: {tcia_root})")

    if not datasets:
        raise RuntimeError(
            "사용 가능한 pretrain 데이터셋이 없습니다. "
            "lits_root 또는 tcia_root를 확인하세요."
        )

    if len(datasets) == 1:
        return datasets[0]
    return CombinedPretrainDataset(datasets)
