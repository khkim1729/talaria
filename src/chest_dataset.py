"""
Chest CT Dataset Placeholder.

ChestCTDataset: target-domain dataset for Chest TNM domain adaptation.

This file is a skeleton — populate sample_list building logic when
chest CT data becomes available.

Expected directory structure (to be confirmed):
    chest_root/
        imagesTr/{case_id}.nii.gz         # CT volume
        labelsTr/{case_id}.nii.gz         # segmentation mask (optional)
        metadata.json                     # maps case_id -> {tstage, nstage}

Label conventions (to be confirmed with data provider):
    0 = background
    1 = lung parenchyma
    2 = primary tumor
    3 = mediastinal lymph node
"""

import os
import json
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Dict

from .dataset import Base3DDataset


class ChestCTDataset(Dataset):
    """
    Chest CT dataset for domain adaptation (target domain).

    When no labels are available (unlabeled=True), returns only images.
    When labels and TNM metadata are available, returns full supervision.

    Args:
        root:       Path to chest CT root directory.
        split:      'train' | 'val' | 'all'
        patch_size: Cubic patch size for sliding window extraction.
        stride:     Sliding window stride.
        transform:  Optional augmentation transform.
        unlabeled:  If True, skip label loading (target domain, no annotations).
        val_ratio:  Fraction of cases used as validation split.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        patch_size: int = 96,
        stride: int = 48,
        transform: Optional[Callable] = None,
        unlabeled: bool = True,
        val_ratio: float = 0.1,
    ):
        self.root       = root
        self.split      = split
        self.patch_size = patch_size
        self.stride     = stride
        self.transform  = transform
        self.unlabeled  = unlabeled

        # Load metadata if present
        meta_path = os.path.join(root, 'metadata.json')
        self.metadata: Dict = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self.sample_list = self._build_sample_list(val_ratio)
        mode = 'pretrain' if unlabeled else 'finetune'
        self._base = Base3DDataset(
            self.sample_list, patch_size, stride, transform, mode
        )

    def _build_sample_list(self, val_ratio: float) -> List[Dict]:
        """
        Scan root/imagesTr for *.nii.gz files and build sample list.
        Splits deterministically by sorted case index.

        TODO: Update path conventions when data structure is confirmed.
        """
        img_dir = os.path.join(self.root, 'imagesTr')
        lbl_dir = os.path.join(self.root, 'labelsTr')

        if not os.path.isdir(img_dir):
            # Data not yet available — return empty list
            return []

        all_cases = sorted(
            f for f in os.listdir(img_dir) if f.endswith('.nii.gz')
        )
        n_val   = max(1, int(len(all_cases) * val_ratio))
        val_set = set(all_cases[-n_val:])

        if self.split == 'train':
            cases = [c for c in all_cases if c not in val_set]
        elif self.split == 'val':
            cases = [c for c in all_cases if c in val_set]
        else:
            cases = all_cases

        samples = []
        for fname in cases:
            case_id = fname.replace('.nii.gz', '')
            img     = os.path.join(img_dir, fname)
            lbl     = os.path.join(lbl_dir, fname)
            meta    = self.metadata.get(case_id, {})
            samples.append({
                'image':  img,
                'label':  lbl if (not self.unlabeled and os.path.exists(lbl)) else None,
                'tstage': meta.get('tstage', -1),
                'nstage': meta.get('nstage', -1),
            })
        return samples

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]
