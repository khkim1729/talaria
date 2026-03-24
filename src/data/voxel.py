import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, CacheDataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer, ControlNetLatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, ControlNet
from generative.networks.schedulers import DDPMScheduler

from monai.data import load_decathlon_datalist


def get_liver_transforms():
    """TNM Staging 학습을 위한 표준 전처리 파이프라인 반환"""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="label"),
        transforms.Resized(
            keys=["image", "label"], 
            spatial_size=(128, 128, 128),
            mode=("bilinear", "nearest") 
        ),
        transforms.ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ])

def get_msd_liver_datalist(data_dir, json_path):
    """MSD Liver 데이터 리스트 로드"""
    return load_decathlon_datalist(
        json_path, 
        is_segmentation=True, 
        data_list_key="training", 
        base_dir=data_dir
    )

