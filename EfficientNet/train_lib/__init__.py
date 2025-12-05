"""
Train/inference library for RSNA IAD.
Modularized helpers to keep the entry script concise.
"""

from .config import CFG, ID_COL, LABEL_COLS, IMAGENET_MEAN, IMAGENET_STD
from .transforms import get_inference_transform, get_train_transform, get_tta_transforms
from .data import process_dicom_series, build_three_channel_image, build_meta_from_metadata, RSNAIADTrainDataset
from .models import MultiBackboneModel, discover_timm_backbone_checkpoint
from .engine import train_one_epoch, validate_one_epoch, predict_single_model, predict_ensemble
from .utils import _seed_everything, _resolve_kaggle_training_paths
