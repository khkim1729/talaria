"""
TALARIA Dataset Classes.

Datasets:
    - LiTSDataset:   LiTS (Liver Tumor Segmentation) — 131 CT scans
                     Labels: 0=background, 1=liver, 2=tumor
    - TCIADataset:   TCIA liver datasets (TCGA-LIHC, CPTAC-LIHC, HCC-TACE-Seg)
                     Labels: tumor masks + optional TNM stage from metadata
    - AMOSDataset:   AMOS (Abdominal Multi-Organ Segmentation) — unlabeled streaming
                     Labels: 15 organ segmentations (used for pre-training only)
    - CombinedDataset: concatenation of all above for Phase 1 pre-training
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Callable, List, Dict, Tuple

from .preprocessing import preprocess_ct, hu_windowing, extract_patches


# ---------------------------------------------------------------------------
# Base 3D CT Dataset
# ---------------------------------------------------------------------------

class Base3DDataset(Dataset):
    """
    Base class for 3D CT patch datasets.
    Loads pre-processed patches (*.npy) from a processed root directory,
    or processes on-the-fly from NIfTI files if processed_root is None.
    """

    def __init__(
        self,
        sample_list: List[Dict],
        patch_size: int = 96,
        stride: int = 48,
        transform: Optional[Callable] = None,
        mode: str = 'pretrain',  # 'pretrain' | 'finetune'
    ):
        """
        Args:
            sample_list: list of dicts with keys:
                         'image'   -> path to .nii.gz CT file
                         'label'   -> path to .nii.gz label file (optional)
                         'tstage'  -> int T-stage label 0-3 (optional)
                         'nstage'  -> int N-stage label 0-1 (optional)
            patch_size:  cubic patch size
            stride:      sliding window stride for patch extraction
            transform:   optional augmentation transform applied to each patch
            mode:        'pretrain' (returns only image) or 'finetune' (returns all)
        """
        self.sample_list = sample_list
        self.patch_size  = patch_size
        self.stride      = stride
        self.transform   = transform
        self.mode        = mode

        # Build flat index: (sample_idx, patch_idx)
        self._index: List[Tuple[int, int]] = []
        self._patches_cache: Dict[int, List[np.ndarray]] = {}
        self._labels_cache:  Dict[int, List[np.ndarray]] = {}

        self._build_index()

    def _build_index(self):
        """Pre-compute patch counts per sample (lazy loading)."""
        for sid, sample in enumerate(self.sample_list):
            # Temporarily load to count patches
            patches, _, _ = preprocess_ct(
                sample['image'], self.patch_size, self.stride
            )
            for pid in range(len(patches)):
                self._index.append((sid, pid))

    def _load_sample(self, sid: int):
        if sid not in self._patches_cache:
            sample = self.sample_list[sid]
            patches, coords, shape = preprocess_ct(
                sample['image'], self.patch_size, self.stride
            )
            self._patches_cache[sid] = patches

            # Load label patches if available
            if 'label' in sample and sample['label'] is not None:
                import SimpleITK as sitk
                from .preprocessing import resample_to_isotropic, extract_patches as ep
                lbl_img = sitk.ReadImage(sample['label'], sitk.sitkUInt8)
                lbl_img = resample_to_isotropic(lbl_img, interpolator=sitk.sitkNearestNeighbor)
                lbl_arr = sitk.GetArrayFromImage(lbl_img).astype(np.float32)
                lbl_patches, _ = ep(lbl_arr, self.patch_size, self.stride)
                self._labels_cache[sid] = lbl_patches
            else:
                self._labels_cache[sid] = None

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        sid, pid = self._index[idx]
        self._load_sample(sid)

        patch = self._patches_cache[sid][pid].copy()
        patch = torch.from_numpy(patch).unsqueeze(0)   # (1, D, H, W)

        sample = self.sample_list[sid]
        result = {'image': patch}

        if self._labels_cache[sid] is not None:
            lbl = self._labels_cache[sid][pid].copy()
            result['label'] = torch.from_numpy(lbl).long()

        if self.mode == 'finetune':
            result['tstage'] = torch.tensor(sample.get('tstage', -1), dtype=torch.long)
            result['nstage'] = torch.tensor(sample.get('nstage', -1), dtype=torch.long)

        if self.transform is not None:
            result['image'] = self.transform(result['image'])

        return result


# ---------------------------------------------------------------------------
# LiTS Dataset
# ---------------------------------------------------------------------------

class LiTSDataset(Dataset):
    """
    LiTS (Liver Tumor Segmentation) Dataset.
    131 abdominal CT scans with liver (label=1) and tumor (label=2) annotations.
    Supports both HCC and ICC histology.

    Expected directory structure:
        lits_root/
            imagesTr/volume-{id}.nii.gz
            labelsTr/segmentation-{id}.nii.gz
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        patch_size: int = 96,
        stride: int = 48,
        transform: Optional[Callable] = None,
        mode: str = 'pretrain',
        val_ids: Optional[List[int]] = None,
    ):
        self.root = root
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.mode = mode

        if val_ids is None:
            val_ids = list(range(121, 131))   # last 10 as validation

        all_ids = list(range(0, 131))
        if split == 'train':
            ids = [i for i in all_ids if i not in val_ids]
        elif split == 'val':
            ids = val_ids
        else:
            ids = all_ids

        self.sample_list = self._build_sample_list(ids)
        self._base = Base3DDataset(self.sample_list, patch_size, stride, transform, mode)

    def _build_sample_list(self, ids):
        samples = []
        for i in ids:
            img_path = os.path.join(self.root, 'imagesTr', f'volume-{i}.nii.gz')
            lbl_path = os.path.join(self.root, 'labelsTr', f'segmentation-{i}.nii.gz')
            if os.path.exists(img_path):
                samples.append({
                    'image': img_path,
                    'label': lbl_path if os.path.exists(lbl_path) else None,
                    'tstage': -1,
                    'nstage': -1,
                })
        return samples

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


# ---------------------------------------------------------------------------
# TCIA Dataset
# ---------------------------------------------------------------------------

class TCIADataset(Dataset):
    """
    TCIA Liver Datasets (TCGA-LIHC, CPTAC-LIHC, HCC-TACE-Seg).
    Optionally loads TNM stage from a metadata JSON file.

    Expected directory structure:
        tcia_root/
            {collection}/
                {case_id}/
                    image.nii.gz
                    label.nii.gz        (optional)
            metadata.json               (optional, maps case_id -> {tstage, nstage})
    """

    def __init__(
        self,
        root: str,
        collections: Optional[List[str]] = None,
        patch_size: int = 96,
        stride: int = 48,
        transform: Optional[Callable] = None,
        mode: str = 'pretrain',
    ):
        self.root = root
        if collections is None:
            collections = ['TCGA-LIHC', 'CPTAC-LIHC', 'HCC-TACE-Seg']
        self.collections = collections

        # Load metadata if available
        meta_path = os.path.join(root, 'metadata.json')
        self.metadata: Dict = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self.sample_list = self._build_sample_list()
        self._base = Base3DDataset(self.sample_list, patch_size, stride, transform, mode)

    def _build_sample_list(self):
        samples = []
        for col in self.collections:
            col_dir = os.path.join(self.root, col)
            if not os.path.isdir(col_dir):
                continue
            for case in sorted(os.listdir(col_dir)):
                case_dir = os.path.join(col_dir, case)
                img = os.path.join(case_dir, 'image.nii.gz')
                lbl = os.path.join(case_dir, 'label.nii.gz')
                if not os.path.exists(img):
                    continue
                meta = self.metadata.get(case, {})
                samples.append({
                    'image':   img,
                    'label':   lbl if os.path.exists(lbl) else None,
                    'tstage':  meta.get('tstage', -1),
                    'nstage':  meta.get('nstage', -1),
                })
        return samples

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


# ---------------------------------------------------------------------------
# AMOS Dataset
# ---------------------------------------------------------------------------

class AMOSDataset(Dataset):
    """
    AMOS (Abdominal Multi-Organ Segmentation) Dataset.
    500 CT + 100 MRI with 15 organ annotations.
    Used primarily for Phase 1 self-supervised pre-training (unlabeled streaming).

    Expected structure (nnUNet format):
        amos_root/
            imagesTr/{case_id}_0000.nii.gz
            labelsTr/{case_id}.nii.gz
    """

    def __init__(
        self,
        root: str,
        modality: str = 'CT',   # 'CT' or 'MRI'
        patch_size: int = 96,
        stride: int = 48,
        transform: Optional[Callable] = None,
        unlabeled: bool = True,  # if True, ignore labels (Phase 1)
    ):
        self.root = root
        self.modality = modality
        self.unlabeled = unlabeled

        self.sample_list = self._build_sample_list()
        mode = 'pretrain' if unlabeled else 'finetune'
        self._base = Base3DDataset(self.sample_list, patch_size, stride, transform, mode)

    def _build_sample_list(self):
        img_dir = os.path.join(self.root, 'imagesTr')
        lbl_dir = os.path.join(self.root, 'labelsTr')
        samples = []
        if not os.path.isdir(img_dir):
            return samples
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('_0000.nii.gz'):
                continue
            case_id = fname.replace('_0000.nii.gz', '')
            img = os.path.join(img_dir, fname)
            lbl = os.path.join(lbl_dir, f'{case_id}.nii.gz')
            samples.append({
                'image': img,
                'label': None if self.unlabeled else (lbl if os.path.exists(lbl) else None),
                'tstage': -1,
                'nstage': -1,
            })
        return samples

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


# ---------------------------------------------------------------------------
# Combined Dataset for Phase 1
# ---------------------------------------------------------------------------

def build_pretrain_dataset(
    lits_root: Optional[str] = None,
    tcia_root: Optional[str] = None,
    amos_root: Optional[str] = None,
    patch_size: int = 96,
    stride: int = 48,
    transform: Optional[Callable] = None,
) -> ConcatDataset:
    """
    Build combined dataset for Phase 1 self-supervised pre-training.
    """
    datasets = []
    if lits_root and os.path.isdir(lits_root):
        datasets.append(LiTSDataset(lits_root, 'train', patch_size, stride, transform, 'pretrain'))
    if tcia_root and os.path.isdir(tcia_root):
        datasets.append(TCIADataset(tcia_root, patch_size=patch_size, stride=stride,
                                    transform=transform, mode='pretrain'))
    if amos_root and os.path.isdir(amos_root):
        datasets.append(AMOSDataset(amos_root, patch_size=patch_size, stride=stride,
                                    transform=transform, unlabeled=True))
    if not datasets:
        raise ValueError("At least one dataset root must be provided and exist.")
    return ConcatDataset(datasets)
