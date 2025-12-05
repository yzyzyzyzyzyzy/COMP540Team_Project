"""
Inference script (RSNA IAD competition example):
- Read DICOM sequences, perform windowing/normalization/sampling and resampling, and construct 3-channel input (middle slice, maximum intensity projection MIP, standard deviation projection).
- Support multiple backbone networks (EffNetV2/ConvNeXt/Swin) and integrate additional meta features (age, gender).
- Support single model or weighted ensemble, TTA, and AMP inference.
- Provide fallback output for abnormal scenarios and clean up disk/VRAM in the finally block to avoid evaluation environment errors.
"""

import os
import sys
import gc
import shutil
import warnings
warnings.filterwarnings('ignore')  # Suppress irrelevant warnings from third-party libraries to keep logs clean
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from IPython.display import display

# Data handling
import numpy as np
import polars as pl
import pandas as pd
import argparse
import time
# Medical imaging
import pydicom  # 读取 DICOM
import cv2      # 图像缩放/颜色空间等

# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast  # AMP 自动混合精度
import timm  # 各种 SOTA 图像骨干网络

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Competition API（评测端要求的接口）
try:
    import kaggle_evaluation.rsna_inference_server as rsna_server
except Exception:
    rsna_server = None

# 优先使用已挂载到 /kaggle/input 的离线缓存，避免联网下载 timm/torch 预训练权重
for _env_name, _candidates in [
    ("TIMM_HOME", ["/kaggle/input/timm-cache", "/kaggle/input/timm", "/kaggle/input/timm_weights"]),
    ("TORCH_HOME", ["/kaggle/input/torch-cache", "/kaggle/input/torch-hub", "/kaggle/input/torch_cache", "/kaggle/input/.cache/torch"]),
]:
    if not os.environ.get(_env_name):
        for _p in _candidates:
            if os.path.isdir(_p):
                os.environ[_env_name] = _p
                print(f"[Offline] Using {_env_name}={_p}")
                break

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# GPU 性能优化开关
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        # 需要 PyTorch >= 2.0，允许更高效的 matmul 精度策略
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


# Competition constants
# 预测输出表头
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# 共享目录（评测环境特定路径）
SHARED_DIR = '/kaggle/shared'

# 常用归一化统计（ImageNet）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 选择推理模型（单模型或集成）
# Options: 'tf_efficientnetv2_s', 'convnext_small', 'swin_small_patch4_window7_224', 'ensemble'
SELECTED_MODEL = 'ensemble' 

# 训练好权重的路径（在 Kaggle 的只读路径下）
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/best/pytorch/default/1/tf_efficientnetv2_s_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class InferenceConfig:
    """
    Inference configuration:
    - Model selection/ensemble
    - Input size, number of slices, windowing
    - Inference batch size, AMP, TTA, number of TTA transforms
    - Weights for ensemble sub-models
    """
    # Model selection
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    
    # 默认输入设置（可被 checkpoint 中的 training_config 覆盖）
    image_size = 512
    num_slices = 32
    use_windowing = True
    
    # Inference settings
    batch_size = 1
    use_amp = True
    use_tta = True
    tta_transforms = 4  # 使用前 4 种 TTA 变换
    
    # Ensemble 权重
    ensemble_weights = {
        'tf_efficientnetv2_s': 1,
        'convnext_small': 0,
        'swin_small_patch4_window7_224': 0
    }

CFG = InferenceConfig()

# ============== Offline timm weight discovery helpers ==============
def _norm_model_key(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name).upper()

def discover_timm_backbone_checkpoint(model_name: str) -> Optional[str]:
    """
    Try to find a local timm pretrained checkpoint for the given model name, without internet:
    - Env override: TIMM_CHECKPOINT or TIMM_CHECKPOINT_<MODEL_KEY>
    - Search TIMM_HOME / TORCH_HOME typical subfolders (checkpoints, weights, hub/checkpoints)
    - Scan /kaggle/input datasets for matching filenames (*model_name*.(pth|pt|bin|safetensors))
    Return a filepath if found, else None.
    """
    # 1) Env overrides
    env_direct = os.environ.get("TIMM_CHECKPOINT")
    if env_direct and os.path.isfile(env_direct):
        return env_direct
    key = f"TIMM_CHECKPOINT_{_norm_model_key(model_name)}"
    env_model = os.environ.get(key)
    if env_model and os.path.isfile(env_model):
        return env_model

    # 2) build candidate directories
    candidates: List[str] = []
    for base in [
        os.environ.get("TIMM_HOME"),
        os.environ.get("TORCH_HOME"),
        "/kaggle/input/timm-cache",
        "/kaggle/input/torch-cache",
        "/kaggle/input/torch-hub",
        "/kaggle/input/timm",
    ]:
        if base and os.path.isdir(base):
            candidates.append(base)
            for sub in ["checkpoints", "hub/checkpoints", "models", "weights"]:
                p = os.path.join(base, sub)
                if os.path.isdir(p):
                    candidates.append(p)

    # 3) scan /kaggle/input one-level
    if os.path.isdir('/kaggle/input'):
        for dn in os.listdir('/kaggle/input'):
            base = os.path.join('/kaggle/input', dn)
            if os.path.isdir(base):
                candidates.append(base)
                for sub in ["checkpoints", "hub/checkpoints", "models", "weights"]:
                    p = os.path.join(base, sub)
                    if os.path.isdir(p):
                        candidates.append(p)

    exts = ('.pth', '.pt', '.bin', '.safetensors')
    name_l = model_name.lower()
    seen = set()
    for root in candidates:
        if not os.path.isdir(root):
            continue
        try:
            for f in os.listdir(root):
                fp = os.path.join(root, f)
                if fp in seen:
                    continue
                seen.add(fp)
                if os.path.isfile(fp) and f.lower().endswith(exts) and name_l in f.lower():
                    return fp
        except Exception:
            continue
    return None

class MultiBackboneModel(nn.Module):
    """Unified wrapper for multiple timm backbones with optional integration of age/gender meta features.

    Args:
        model_name (str): timm 模型名。
        num_classes (int): 输出类别数。
        pretrained (bool): 是否加载 timm 预训练（若给定 backbone_checkpoint_path 则忽略）。
        drop_rate (float): 分类头 dropout 比例。
        drop_path_rate (float): backbone stochastic depth 比例。
        freeze_backbone (bool): 是否冻结骨干网络参数。
        backbone_checkpoint_path (Optional[str]): 离线骨干权重路径。
        unfreeze_last_n_layers (int): 冻结时可部分解冻最后 N 个顶层子模块。
        use_meta_features (bool): 是否启用元特征分支（默认 False）。
    """
    def __init__(self, model_name: str, num_classes: int = 14, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2, freeze_backbone: bool = False,
                 backbone_checkpoint_path: Optional[str] = None, unfreeze_last_n_layers: int = 0,
                 use_meta_features: bool = False) -> None:
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.use_meta_features = bool(use_meta_features)
        
        # 创建 backbone（去除分类头与全局池化，输出“特征”而不是 logits）
        if 'swin' in model_name:
            # Swin 默认 224×224，这里通过 img_size 参数覆盖，适配我们统一的 CFG.image_size
            self.backbone = timm.create_model(
                model_name, 
                pretrained=(pretrained and backbone_checkpoint_path is None),
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,  # 覆盖默认输入分辨率
                num_classes=0,  # 去掉分类头
                global_pool=''  # 去掉全局池化，由我们手动处理
            )
        else:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=(pretrained and backbone_checkpoint_path is None),
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,  # 去掉分类头
                global_pool=''  # 去掉全局池化
            )
        # If local backbone checkpoint provided, load with strict=False
        if backbone_checkpoint_path and os.path.isfile(backbone_checkpoint_path):
            try:
                print(f"[Offline] Loading backbone weights from {backbone_checkpoint_path}")
                ckpt = torch.load(backbone_checkpoint_path, map_location='cpu')
                state = None
                if isinstance(ckpt, dict):
                    # common containers
                    for k in ['state_dict', 'model', 'weights', 'params']:
                        v = ckpt.get(k)
                        if isinstance(v, dict):
                            state = v
                            break
                    if state is None:
                        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)} or ckpt
                else:
                    state = ckpt
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                if isinstance(missing, list) and isinstance(unexpected, list):
                    if missing or unexpected:
                        print(f"[Offline] Loaded with missing={len(missing)} unexpected={len(unexpected)} keys (strict=False)")
            except Exception as e:
                print(f"[Offline] Failed to load backbone checkpoint: {e}")
        # 可选：冻结骨干参数
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # 部分解冻：仅解冻最后 N 个顶层子模块（children）
            if self.unfreeze_last_n_layers > 0:
                children = list(self.backbone.children())
                if children:
                    for layer in children[-self.unfreeze_last_n_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                    print(f"[Unfreeze] Last {self.unfreeze_last_n_layers} backbone layers unfrozen.")

            # 默认策略（测试/微调友好）：EfficientNetV2 选择性解冻最后3个blocks + conv_head + bn2
            # 仅当未显式指定 unfreeze_last_n_layers 时启用
            if self.unfreeze_last_n_layers == 0 and ('efficientnetv2' in self.model_name or 'tf_efficientnetv2' in self.model_name):
                try:
                    # 计算 blocks 的总数，选择最后3个
                    # timm EfficientNetV2 结构通常包含属性 self.backbone.blocks (list/Sequential)
                    last_block_indices = []
                    if hasattr(self.backbone, 'blocks'):
                        try:
                            n_blocks = len(self.backbone.blocks)
                            last_block_indices = list(range(max(0, n_blocks - 3), n_blocks))
                        except Exception:
                            last_block_indices = []

                    restored = 0
                    for name, param in self.backbone.named_parameters():
                        # 解冻 conv_head 与 bn2
                        if name.startswith('conv_head') or name.startswith('bn2'):
                            param.requires_grad = True
                            restored += 1
                            continue
                        # 解冻最后三个 blocks
                        if name.startswith('blocks.') and last_block_indices:
                            # name like 'blocks.7.xxx'; 解析索引
                            try:
                                parts = name.split('.')
                                idx = int(parts[1])
                                if idx in last_block_indices:
                                    param.requires_grad = True
                                    restored += 1
                            except Exception:
                                pass
                    print(f"[Selective Unfreeze] EfficientNetV2 restored params: {restored} (last 3 blocks + conv_head + bn2)")
                except Exception as e:
                    print(f"[Selective Unfreeze] Skipped due to error: {e}")
            # 精细化：EfficientNetV2 只解冻末尾 blocks + conv_head
            if 'efficientnetv2' in self.model_name:
                # 若不想用 unfreeze_last_n_layers，手动挑选
                selective = ['blocks.6', 'blocks.7', 'conv_head', 'bn2']  # 视具体版本调整
                restored = 0
                for name, p in self.backbone.named_parameters():
                    for key in selective:
                        if name.startswith(key):
                            p.requires_grad = True
                            restored += 1
                            break
                if restored > 0:
                    print(f"[Selective Unfreeze] EfficientNetV2 params restored={restored} ({selective})")
        # 运行一次 dummy 前向来“探测”backbone 输出的形状，从而决定后续如何池化/展平
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                # 卷积特征：(B, C, H, W) -> 需要 2D 全局池化
                num_features = features.shape[1]
                self.needs_pool = True
                self.needs_seq_pool = False
            elif len(features.shape) == 3:
                # Transformer 特征：(B, N, D) -> 需要对 token 维做平均池化
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # 已是扁平特征：(B, D)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")
        
        # 若是卷积输出，则加一个自适应平均池化层
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 元特征（年龄、性别共 2 维）的 MLP 处理分支（仅在 use_meta_features=True 时有效参与）
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        meta_dim = 32 if self.use_meta_features else 0
        
        # 分类头：拼接图像特征与元特征后，接 512->256->num_classes
        # 使用 BatchNorm 提升稳定性，Dropout 控制过拟合
        self.classifier = nn.Sequential(
            nn.Linear(num_features + meta_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。

        Args:
            image: (B,3,H,W)
            meta:  (B,2) 或 None，当 use_meta_features=False 时忽略。
        Returns:
            (B,num_classes) logits
        """
        # 1) 提取图像特征（冻结时禁用梯度，节省显存）
        if getattr(self, 'freeze_backbone', False):
            with torch.no_grad():
                img_features = self.backbone(image)
        else:
            img_features = self.backbone(image)

        # 2) 根据骨干输出形状做合适的池化/展平
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # 卷积输出：全局平均池化 -> (B, C, 1, 1) -> (B, C)
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer 输出：对 token 维取均值 -> (B, D)
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            # 兜底：任何 4D 输出都做一次自适应池化
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            # 兜底：任何 3D 输出对 dim=1 做均值
            img_features = img_features.mean(dim=1)

        # 3) 可选处理元特征
        if self.use_meta_features and meta is not None:
            meta_features = self.meta_fc(meta)
            combined = torch.cat([img_features, meta_features], dim=1)
        else:
            combined = img_features

        # 4) 分类
        return self.classifier(combined)
    

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    按 DICOM 窗宽/窗位进行裁剪与线性缩放到 [0, 255] 的 uint8。
    - img: 原始像素（已应用 RescaleSlope/Intercept）
    - window_center/window_width: 窗位/窗宽
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    """
    根据模态给出经验性的窗宽窗位参数（示例值，可按实际任务调整）
    - 返回 (window_center, window_width)
    """
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350),
        'MRA': (600, 1200),
        'MRI': (40, 80),
    }
    return windows.get(modality, (40, 80))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    将任意浮点矩阵线性归一化到 [0, 255] 的 uint8；若常数阵则返回全零。
    """
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
        return (arr * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def build_three_channel_image(volume: np.ndarray) -> np.ndarray:
    """
    从 (D,H,W) 的体数据构造 H×W×3 的输入图像：
    - 通道1：中间层
    - 通道2：最大强度投影（MIP）
    - 通道3：标准差投影（STD，经 _normalize_to_uint8 归一化）
    """
    middle_slice = volume[CFG.num_slices // 2]
    mip = np.max(volume, axis=0)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    std_proj = _normalize_to_uint8(std_proj)
    return np.stack([middle_slice, mip, std_proj], axis=-1)


def build_meta_from_metadata(metadata: Dict) -> np.ndarray:
    """将字典元数据转为网络需要的 2 维向量 [age_norm, sex]。"""
    age_normalized = float(metadata.get('age', 50)) / 100.0
    sex = float(metadata.get('sex', 0))
    return np.array([age_normalized, sex], dtype=np.float32)

def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    """
    处理一个 DICOM 序列目录：
    - 遍历目录内所有 .dcm
    - 读取像素并应用 RescaleSlope/Intercept
    - 窗口化或最小-最大归一化到 uint8
    - resize 到 CFG.image_size
    - 采样/补齐到 CFG.num_slices 张切片
    返回：
    - volume: (num_slices, H, W) 的 uint8 体数据
    - metadata: {'age': int, 'sex': 0/1, 'modality': str}
    """
    series_path = Path(series_path)
    
    # 递归收集所有 DICOM 文件路径
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    # 可选快速路径：当 DICOM 数量远大于目标切片数时，仅采样读取一部分文件以降低 I/O
    # 启用方式：设置环境变量 FAST_READ=1（默认关闭，完整读取）
    fast_read = os.environ.get("FAST_READ", "0") == "1"
    if fast_read and len(all_filepaths) > CFG.num_slices:
        idx = np.linspace(0, len(all_filepaths) - 1, CFG.num_slices).astype(int)
        sampled = [all_filepaths[i] for i in idx]
        if len(sampled) != len(all_filepaths):
            print(f"[FastRead] Sampling {len(sampled)}/{len(all_filepaths)} DICOMs for speed.")
        all_filepaths = sampled

    if len(all_filepaths) == 0:
        # 找不到 DICOM，返回全零体积与默认元特征（保证管线可运行）
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata
    
    # 逐张读取与处理
    slices = []
    metadata = {}
    
    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # 处理多帧或彩色图像：
            # - 彩色：转灰度
            # - 多帧：取第 0 帧
            if img.ndim == 3:
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    img = img[:, :, 0]
            
            # 第 1 张图提取元数据
            if i == 0:
                metadata['modality'] = getattr(ds, 'Modality', 'CT')
                
                # 年龄字段通常是 '050Y'，此处解析前三位数字
                try:
                    age_str = getattr(ds, 'PatientAge', '050Y')
                    age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
                    metadata['age'] = min(age, 100)
                except:
                    metadata['age'] = 50
                
                # 性别：默认 'M' -> 1，其他 -> 0
                try:
                    sex = getattr(ds, 'PatientSex', 'M')
                    metadata['sex'] = 1 if sex == 'M' else 0
                except:
                    metadata['sex'] = 0
            
            # 应用 RescaleSlope/Intercept（CT 常见）
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            # 窗口化或 min-max 归一化
            if CFG.use_windowing:
                window_center, window_width = get_windowing_params(metadata['modality'])
                img = apply_dicom_windowing(img, window_center, window_width)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            # resize 成统一分辨率
            img = cv2.resize(img, (CFG.image_size, CFG.image_size))
            slices.append(img)
            
        except Exception as e:
            # 单张失败不致命，跳过并继续
            print(f"Error processing {filepath}: {e}")
            continue
    
    # 按需采样/补齐到固定的切片数 CFG.num_slices
    if len(slices) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    else:
        volume = np.array(slices)  # (D, H, W)
        if len(slices) > CFG.num_slices:
            # 线性等间距抽取 CFG.num_slices 张
            indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
            volume = volume[indices]
        elif len(slices) < CFG.num_slices:
            # 用边缘复制的方式在“深度维”补齐
            pad_size = CFG.num_slices - len(slices)
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    return volume, metadata

def get_inference_transform() -> A.Compose:
    """
    基础推理变换：Normalize 到 ImageNet 统计并转 tensor。
    注意：这里假设 3 通道输入（我们构造的 middle/MIP/STD）。
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_tta_transforms() -> List[A.Compose]:
    """
    TTA（测试时增强）变换：
    - 原图
    - 水平翻转
    - 垂直翻转
    - 90 度旋转
    最后统一再做 Normalize + ToTensorV2
    """
    transforms = [
        A.Compose([  # Original
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([  # Horizontal flip
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([  # Vertical flip
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]),
        A.Compose([  # 90 degree rotation
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    ]
    return transforms

# 全局对象：已加载模型与变换
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """
    加载一个单模型：
    - 反序列化 checkpoint（允许 numpy 标量，因此 weights_only=False）
    - 从 checkpoint 读取 training_config 覆盖部分 CFG（如 image_size）
    - 按模型名创建 MultiBackboneModel，并加载权重字典 'model_state_dict'
    """
    print(f"Loading {model_name} from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 注意：weights_only=False 以兼容包含 numpy 标量的 checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 可选的配置（训练时保存）
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    # 若训练时保存了 image_size，则以训练配置为准，避免尺寸不匹配
    if 'image_size' in training_config:
        CFG.image_size = training_config['image_size']
    
    # 初始化骨干模型（pretrained=False，纯加载我们保存的权重）
    model = MultiBackboneModel(
        model_name=model_name,
        num_classes=training_config.get('num_classes', 14),
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best = checkpoint.get('best_score', None)
    if isinstance(best, (int, float)):
        print(f"Loaded {model_name} with best score: {best:.4f}")
    else:
        print(f"Loaded {model_name}")
     
    return model

def load_models() -> None:
    """
    根据配置加载模型：
    - 如果 use_ensemble=True，按 MODEL_PATHS 逐个加载，允许单个失败不终止
    - 否则仅加载选定的单模型
    - 初始化基础变换和 TTA 变换
    - 用随机输入做一次“热身”，确保首次推理不抖动
    """
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    print("Loading models...")
    
    if CFG.use_ensemble:
        # 加载集成中的各模型
        for model_name, model_path in MODEL_PATHS.items():
            try:
                MODELS[model_name] = load_single_model(model_name, model_path)
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
    else:
        # 加载单模型
        if CFG.model_selection in MODEL_PATHS:
            model_path = MODEL_PATHS[CFG.model_selection]
            MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
        else:
            raise ValueError(f"Unknown model: {CFG.model_selection}")
    
    # 初始化变换
    TRANSFORM = get_inference_transform()
    if CFG.use_tta:
        TTA_TRANSFORMS = get_tta_transforms()
    
    print(f"Models loaded: {list(MODELS.keys())}")
    
    # 热身：跑一次前向，编译 cudnn 算子，避免首帧延迟
    print("Warming up models...")
    dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_meta = torch.randn(1, 2).to(device)
    
    with torch.no_grad():
        for model in MODELS.values():
            _ = model(dummy_image, dummy_meta)
    
    print("Ready for inference!")


def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """
    对单个模型做一次预测：
    - 支持 TTA：对若干几何增强后的图片分别前向，结果做平均
    - 使用 autocast 做 AMP 混合精度推理
    参数：
      image: H×W×3 的 uint8 图像（middle/MIP/STD）
      meta_tensor: (1,2) 的元特征张量
    返回：
      (14,) 的概率（sigmoid 后）
    """
    predictions = []
    
    if CFG.use_tta and TTA_TRANSFORMS:
        # Test time augmentation
        for transform in TTA_TRANSFORMS[:CFG.tta_transforms]:
            aug_image = transform(image=image)['image']  # (3,H,W) float tensor
            aug_image = aug_image.unsqueeze(0).to(device)  # (1,3,H,W)
            
            with torch.no_grad():
                with autocast(enabled=CFG.use_amp):
                    output = model(aug_image, meta_tensor)       # (1,14) logits
                    pred = torch.sigmoid(output)                 # 概率
                    predictions.append(pred.cpu().numpy())
        
        # 平均 TTA 结果
        return np.mean(predictions, axis=0).squeeze()
    else:
        # 单次预测
        image_tensor = TRANSFORM(image=image)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                output = model(image_tensor, meta_tensor)
                return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_ensemble(image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """
    集成预测：对已加载的多个模型分别预测，并按 CFG.ensemble_weights 做加权平均。
    返回 (14,) 概率。
    """
    all_predictions = []
    weights = []
    
    for model_name, model in MODELS.items():
        pred = predict_single_model(model, image, meta_tensor)
        all_predictions.append(pred)
        weights.append(CFG.ensemble_weights.get(model_name, 1.0))
    
    # 归一化权重并加权平均
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    
    return np.average(predictions, weights=weights, axis=0)

def _predict_inner(series_path: str) -> pl.DataFrame:
    """
    核心推理逻辑：
    - 懒加载模型
    - 处理 DICOM 得到 3D 体数据与元特征
    - 构造 3 通道 2D 输入（中间层、MIP、STD），以及归一化的 meta tensor
    - 单模型或集成预测 -> Polars DataFrame（无 ID 列，符合评测端接口）
    """
    global MODELS
    
    # 懒加载模型
    if not MODELS:
        load_models()
    
    # 当前序列 ID（目录名）
    series_id = os.path.basename(series_path)
    
    # 预处理 DICOM 序列 -> 体数据 (D,H,W) 与元数据
    volume, metadata = process_dicom_series(series_path)
    
    # 将 3D 体投影为 3 通道 2D：
    # H×W×3（三通道）输入 + 元特征
    image = build_three_channel_image(volume)
    meta_np = build_meta_from_metadata(metadata)
    meta_tensor = torch.tensor(meta_np[None, :], dtype=torch.float32).to(device)
    
    # 推理
    if CFG.use_ensemble:
        final_pred = predict_ensemble(image, meta_tensor)
    else:
        model = MODELS[CFG.model_selection]
        final_pred = predict_single_model(model, image, meta_tensor)
    
    # 组装输出（注意评测端需要不带 ID 列的列序）
    predictions_df = pl.DataFrame(
        data=[[series_id] + final_pred.tolist()],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )

    # 返回时丢弃 ID 列（Kaggle 接口会自己填）
    return predictions_df.drop(ID_COL)


def predict_fallback(series_path: str) -> pl.DataFrame:
    """
    发生异常时的兜底预测：
    - 返回固定的“小概率”分数，保持输出 schema 正确
    - 清理共享目录，避免评测端磁盘/目录残留问题
    """
    series_id = os.path.basename(series_path)
    
    predictions = pl.DataFrame(
        data=[[series_id] + [0.1] * len(LABEL_COLS)],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )
    
    # 清理共享目录（评测环境约定）
    shutil.rmtree(SHARED_DIR, ignore_errors=True)
    
    return predictions.drop(ID_COL)

def predict(series_path: str) -> pl.DataFrame:
    """
    评测端要求暴露的顶层函数：
    - try: 调用核心逻辑 _predict_inner
    - except: 输出 fallback（同时打印错误，方便排查）
    - finally: 始终清空 /kaggle/shared，且清空 CUDA cache / 触发 GC，防止“磁盘不足/目录非空/显存泄露”
    """
    try:
        # 正常推理
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        # 返回与接口列完全一致的 DataFrame（无 ID 列）
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # 必须的清理逻辑：评测环境会在 /kaggle/shared 写入/缓存中间文件，需清空并重建
        shutil.rmtree(SHARED_DIR, ignore_errors=True)
        os.makedirs(SHARED_DIR, exist_ok=True)
        
        # 显存与内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    

# ============== 训练与验证（从零训练） ==============

from torch.utils.data import Dataset, DataLoader

def _resolve_kaggle_training_paths(train_csv: Optional[str], data_root: Optional[str]) -> Tuple[str, str]:
    """
    训练路径解析器（适配 Kaggle 输入结构）：
    - 支持以下传参形式：
      1) train_csv 为文件，data_root 为 series 目录
      2) train_csv 为“数据集根目录”（包含 train.csv 与 series），data_root 为空或为根目录
      3) 仅传 train_csv 或仅传 data_root，另一路径自动从同级目录推断
      4) 两者都为空或无效：自动扫描 /kaggle/input 下第一个包含 train.csv 与 series 的数据集
    返回：
      (train_csv_path, series_root_path)
    """
    def is_series_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.basename(os.path.normpath(p)).lower() == 'series'
    
    # 1) 若 train_csv 是目录，则视作“数据集根”，拼接出真正的 train.csv
    if train_csv and os.path.isdir(train_csv):
        candidate = os.path.join(train_csv, 'train.csv')
        if os.path.isfile(candidate):
            train_csv = candidate
 
    # 2) 若 data_root 是根目录（非 series），尝试附加 /series
    if data_root and os.path.isdir(data_root) and not is_series_dir(data_root):
        series_candidate = os.path.join(data_root, 'series')
        if os.path.isdir(series_candidate):
            data_root = series_candidate
 
    # 3) 若只给了 train_csv 文件，则从其父目录推断 series
    if (not data_root or not os.path.isdir(data_root)) and train_csv and os.path.isfile(train_csv):
        root = os.path.dirname(os.path.abspath(train_csv))
        series_candidate = os.path.join(root, 'series')
        if os.path.isdir(series_candidate):
            data_root = series_candidate
 
    # 4) 若只给了 data_root（series 或其父目录），推断 train.csv
    if (not train_csv or not os.path.isfile(train_csv)) and data_root and os.path.isdir(data_root):
        root = os.path.abspath(data_root)
        if is_series_dir(root):
            root = os.path.dirname(root)
        csv_candidate = os.path.join(root, 'train.csv')
        if os.path.isfile(csv_candidate):
            train_csv = csv_candidate
 
    # 5) 兜底：扫描 /kaggle/input
    if (not train_csv or not os.path.isfile(train_csv)) or (not data_root or not os.path.isdir(data_root)):
        kaggle_input = '/kaggle/input'
        if os.path.isdir(kaggle_input):
            for name in os.listdir(kaggle_input):
                base = os.path.join(kaggle_input, name)
                csv_candidate = os.path.join(base, 'train.csv')
                series_candidate = os.path.join(base, 'series')
                if os.path.isfile(csv_candidate) and os.path.isdir(series_candidate):
                    train_csv = csv_candidate
                    data_root = series_candidate
                    break
 
    # 最终校验
    if not train_csv or not os.path.isfile(train_csv):
        raise FileNotFoundError(f"未找到 train.csv，请检查传入路径或 Kaggle 数据集挂载。train_csv={train_csv}")
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(f"未找到 series 目录，请检查传入路径或 Kaggle 数据集挂载。data_root={data_root}")
 
    print(f"[Paths] train_csv = {train_csv}")
    print(f"[Paths] series_root = {data_root}")
    return train_csv, data_root

def get_train_transform():
    """
    训练增强：轻量几何/颜色扰动 + Normalize + ToTensorV2
    说明：process_dicom_series 已经 resize 到 CFG.image_size，这里不再改变分辨率。
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class RSNAIADTrainDataset(Dataset):
    """
    训练集数据集：
    - df: pandas.DataFrame，包含 [ID_COL] + LABEL_COLS 列
    - root_dir: DICOM 序列根目录（每个 UID 一个子目录）
    - transform: Albumentations 变换（返回 tensor）
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None, cache_dir: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        # 不使用磁盘缓存，直接在线处理 DICOM（更简洁）

    def __len__(self):
        return len(self.df)

    def _make_image_and_meta(self, series_path: str):
        # 利用现有的序列处理函数
        volume, metadata = process_dicom_series(series_path)

        middle_slice = volume[CFG.num_slices // 2]
        mip = np.max(volume, axis=0)
        std_proj = np.std(volume, axis=0).astype(np.float32)
        if std_proj.max() > std_proj.min():
            std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
        else:
            std_proj = np.zeros_like(std_proj, dtype=np.uint8)

        image = np.stack([middle_slice, mip, std_proj], axis=-1)  # H W 3

        age_normalized = float(metadata.get('age', 50)) / 100.0
        sex = float(metadata.get('sex', 0))
        meta = np.array([age_normalized, sex], dtype=np.float32)
        return image, meta

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[ID_COL])
        series_path = os.path.join(self.root_dir, uid)
        # 直接在线处理
        image, meta = self._make_image_and_meta(series_path)
        if self.transform is not None:
            image = self.transform(image=image)['image']  # (3,H,W) tensor [float]
        else:
            # 与推理同样的标准化
            image = get_inference_transform()(image=image)['image']

        # 目标多标签
        target = row[LABEL_COLS].astype(np.float32).values
        target = torch.from_numpy(target)

        meta_tensor = torch.from_numpy(meta)  # (2,)
        return image, meta_tensor, target



def _seed_everything(seed: int = 42):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """
    计算 BCEWithLogitsLoss 的 pos_weight，缓解类别不平衡。
    pos_weight = (N - P) / (P + eps)
    """
    eps = 1e-6
    y = train_df[LABEL_COLS].values.astype(np.float32)
    P = y.sum(axis=0)
    N = y.shape[0] - P
    w = (N / (P + eps)).astype(np.float32)
    w[np.isinf(w)] = 1.0
    w[np.isnan(w)] = 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, use_amp=True, log_interval=10):
    model.train()
    running_loss = 0.0
    count = 0
    last_end = time.time()
    log_gpu = os.environ.get('LOG_GPU', '0') == '1'
    # 可通过环境变量或上层传参限制每轮的最大步数（快速调试/微调）
    max_steps_env = os.environ.get('MAX_STEPS_PER_EPOCH')
    max_steps = int(max_steps_env) if max_steps_env and max_steps_env.isdigit() else None

    for step, (images, metas, targets) in enumerate(loader):
        # 数据到达的时间（粗略视为 data loading 时间）
        now = time.time()
        data_time = now - last_end
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        comp_start = time.time()
        with autocast(enabled=use_amp):
            logits = model(images, metas)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        comp_time = time.time() - comp_start
        last_end = time.time()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

        if (step + 1) % log_interval == 0:
            if log_gpu and torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024**2)
                mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"Epoch {epoch} | Step {step+1}/{len(loader)} | Loss {loss.item():.4f} | data {data_time:.3f}s | comp {comp_time:.3f}s | mem {mem:.1f}MB (peak {mem_peak:.1f}MB)")
            else:
                print(f"Epoch {epoch} | Step {step+1}/{len(loader)} | Loss {loss.item():.4f} | data {data_time:.3f}s | comp {comp_time:.3f}s")

        # 早停本轮，限制步数（用于快速微调）
        if max_steps is not None and (step + 1) >= max_steps:
            print(f"[FastTune] Stop early at step {step+1} / {len(loader)} for epoch {epoch} (MAX_STEPS_PER_EPOCH={max_steps}).")
            break

    return running_loss / max(1, count)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    for images, metas, targets in loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=False):  # 评估期不必 AMP
            logits = model(images, metas)
            loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / max(1, count)


def train_model(
    train_csv: str,
    data_root: str,
    out_dir: str,
    model_name: str = 'tf_efficientnetv2_s',
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    val_ratio: float = 0.2,
    seed: int = 42,
    pretrained_backbone: bool = True,   # 使用 timm 预训练
    freeze_backbone: bool = True,       # 只训练最后几层与 meta 分支
    unfreeze_last_n_layers: int = 0,    # 若 freeze_backbone=True, 仍解冻最后 N 个顶层子模块
    num_workers: int = 2,
    use_amp: bool = True,
    # Fast-tune options
    sample_frac: Optional[float] = None,   # 按比例抽样（0-1]，用于快速微调
    max_train: Optional[int] = None,       # 训练集最多多少样本（二选一，优先 sample_frac）
    max_val: Optional[int] = None,         # 验证集最多多少样本
    image_size: Optional[int] = None,      # 覆盖输入分辨率（例如 384/320 加速）
    num_slices: Optional[int] = None,      # 覆盖切片数（例如 16 加速）
    enable_fast_read: bool = False,        # 仅抽样读取需要的 DICOM 文件，降低 I/O
    max_steps_per_epoch: Optional[int] = None,  # 每轮最多训练多少步（用于快速验证管道）
):
    """
    End-to-end training:
    - train_csv: Can pass "file path" or "dataset root directory" (containing train.csv and series)
    - data_root: Can pass "series directory" or "dataset root directory", both can be automatically corrected
    - out_dir: Model output directory (save best.pth)
    """
    os.makedirs(out_dir, exist_ok=True)
    _seed_everything(seed)
    # 统一解析路径，自动适配 Kaggle 输入结构
    train_csv, data_root = _resolve_kaggle_training_paths(train_csv, data_root)
 
    # Fast-read 开关（减少单样本 I/O）
    if enable_fast_read:
        os.environ['FAST_READ'] = '1'
        print('[FastTune] FAST_READ enabled: only reading sampled DICOM files per series.')
    # 覆盖输入尺度/切片数（减少单样本工作量）
    if image_size is not None:
        CFG.image_size = int(image_size)
        print(f"[FastTune] Override image_size = {CFG.image_size}")
    if num_slices is not None:
        CFG.num_slices = int(num_slices)
        print(f"[FastTune] Override num_slices = {CFG.num_slices}")

    # 使用 pandas 读取标签
    df = pd.read_csv(train_csv)
    # 简单分层：如果存在 'Aneurysm Present' 列则按该列分层
    if 'Aneurysm Present' in df.columns:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx_tr, idx_va = next(sss.split(df, df['Aneurysm Present'].values))
    else:
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx_tr, idx_va = next(ss.split(df))

    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_va = df.iloc[idx_va].reset_index(drop=True)

    # 子样本抽样函数
    def _subsample(df_in: pd.DataFrame, frac: Optional[float], max_n: Optional[int], seed_local: int) -> pd.DataFrame:
        if df_in.empty:
            return df_in
        # 优先按比例抽样
        if frac is not None and 0 < frac < 1:
            if 'Aneurysm Present' in df_in.columns:
                g = df_in.groupby('Aneurysm Present', group_keys=False)
                return g.apply(lambda d: d.sample(frac=min(frac, 1.0), random_state=seed_local))
            else:
                return df_in.sample(frac=min(frac, 1.0), random_state=seed_local)
        # 否则按数量抽样
        if max_n is not None and max_n > 0 and max_n < len(df_in):
            if 'Aneurysm Present' in df_in.columns:
                counts = df_in['Aneurysm Present'].value_counts()
                total = len(df_in)
                parts = {}
                # 按比例为各组分配样本数，至少 1（若该组存在）
                for k, cnt in counts.items():
                    n_k = max(1, int(round(cnt / total * max_n)))
                    parts[k] = min(n_k, cnt)
                frames = []
                for k, n_k in parts.items():
                    frames.append(df_in[df_in['Aneurysm Present'] == k].sample(n=n_k, random_state=seed_local))
                out = pd.concat(frames).sample(frac=1.0, random_state=seed_local).reset_index(drop=True)
                return out
            else:
                return df_in.sample(n=max_n, random_state=seed_local).reset_index(drop=True)
        return df_in

    # 应用快速抽样：默认仅对训练集抽样，验证集保持完整（除非显式传入 max_val）
    df_tr = _subsample(df_tr, sample_frac, max_train, seed)
    df_va = _subsample(df_va, None, max_val, seed)
    print(f"[FastTune] After subsample -> Train size = {len(df_tr)} | Val size = {len(df_va)}")

    # 数据集与加载器（不使用磁盘缓存与预处理数据集）
    train_ds = RSNAIADTrainDataset(df_tr, data_root, transform=get_train_transform())
    val_ds = RSNAIADTrainDataset(df_va, data_root, transform=get_inference_transform())

    print(f"[Data] Train size = {len(train_ds)} | Val size = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None)
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size // 2), shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None)
    )

    # 配置全局 CFG 以保证训练/推理一致（可按需覆盖）
    CFG.model_selection = model_name
    CFG.use_ensemble = False

    # 模型
    backbone_ckpt = None
    if pretrained_backbone:
        backbone_ckpt = discover_timm_backbone_checkpoint(model_name)
        if backbone_ckpt:
            print(f"[Offline] Found local pretrained backbone for {model_name}: {backbone_ckpt}")
        else:
            print(f"[Offline] No local pretrained weights found for {model_name}. Proceeding with random init (pretrained_backbone=False).")
            pretrained_backbone = False

    model = MultiBackboneModel(
        model_name=model_name,
        num_classes=len(LABEL_COLS),
        pretrained=pretrained_backbone,
        drop_rate=0.2,
        drop_path_rate=0.1,
        freeze_backbone=freeze_backbone,
        backbone_checkpoint_path=backbone_ckpt,
        unfreeze_last_n_layers=unfreeze_last_n_layers
    ).to(device)

    # 损失/优化器/调度器
    pos_weight = _make_pos_weight(df_tr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if freeze_backbone and unfreeze_last_n_layers == 0:
        # 仅训练分类头 + meta 分支
        head_params = list(model.classifier.parameters()) + list(model.meta_fc.parameters())
        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
    else:
        # 训练所有 requires_grad=True 的参数（包含部分解冻的层）
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_path = os.path.join(out_dir, f'{model_name}_best.pth')
    last_path = os.path.join(out_dir, f'{model_name}_last.pth')
    print(f"[Save] Best checkpoint target: {best_path}")
    print(f"[Save] Last checkpoint target: {last_path}")

    # 若指定了每轮最大步数，设置到环境变量，方便 train_one_epoch 读取
    if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
        os.environ['MAX_STEPS_PER_EPOCH'] = str(int(max_steps_per_epoch))
        print(f"[FastTune] Limit steps per epoch = {max_steps_per_epoch}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, epoch, use_amp=use_amp)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        # 始终保存 last 以便调试/复现（不会覆盖 best 的选择逻辑）
        try:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_score': float(-min(best_val_loss, val_loss)),
                        'training_config': {
                            'image_size': CFG.image_size,
                            'num_classes': len(LABEL_COLS),
                            'num_slices': CFG.num_slices,
                            'use_windowing': CFG.use_windowing,
                            'model_name': model_name,
                        }}, last_path)
        except Exception as e:
            print(f"[Warn] Failed to save last checkpoint: {e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': float(-best_val_loss),  # 兼容已有打印，越大越好
                'training_config': {
                    'image_size': CFG.image_size,
                    'num_classes': len(LABEL_COLS),
                    'num_slices': CFG.num_slices,
                    'use_windowing': CFG.use_windowing,
                    'model_name': model_name,
                },
                'model_config': {
                    'drop_rate': 0.2,
                    'drop_path_rate': 0.1,
                }
            }
            torch.save(checkpoint, best_path)
            print(f"Saved new best to {best_path}")

    print(f"Training finished. Best val_loss={best_val_loss:.4f}. Best checkpoint: {best_path}")
    if not os.path.isfile(best_path):
        print(f"[Warn] Best checkpoint not found on disk. Check above logs and ensure dataset/epochs ran. Last checkpoint (if any): {last_path if os.path.isfile(last_path) else 'N/A'}")
    return best_path




def use_local_checkpoint(model_name: str, model_path: str):
    """
    将本地训练得到的权重注册到本脚本的推理管线：
    - 更新 MODEL_PATHS
    - 切换为单模型推理
    - 清空已加载的 MODELS（下次预测时懒加载）
    使用示例：
        best = train_model(...); use_local_checkpoint('tf_efficientnetv2_s', best)
        preds = predict(series_path)
    """
    MODEL_PATHS[model_name] = model_path
    CFG.model_selection = model_name
    CFG.use_ensemble = False
    MODELS.clear()
    print(f"Registered local checkpoint for {model_name}: {model_path}")

def prepare_trained_model_for_serving(
    train_csv: Optional[str] = None,
    data_root: Optional[str] = None,
    out_dir: str = "/kaggle/working/out",
    model_name: str = "tf_efficientnetv2_s",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    val_ratio: float = 0.2,
    seed: int = 42,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    unfreeze_last_n_layers: int = 0,
    num_workers: int = 0,
    use_amp: bool = True,
    reuse_if_exists: bool = True,
    # Fast-tune passthrough
    sample_frac: Optional[float] = None,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    image_size: Optional[int] = None,
    num_slices: Optional[int] = None,
    enable_fast_read: bool = False,
    max_steps_per_epoch: Optional[int] = None,
):
    """
    训练模型并注册到推理管线；若已有 best.pth 且允许复用，则跳过训练直接注册。
    返回：best 权重路径。
    """
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{model_name}_best.pth")

    if reuse_if_exists and os.path.isfile(best_path):
        print(f"[Serve] Found existing checkpoint, reuse: {best_path}")
    else:
        print("[Serve] Training model before serving...")
        best_path = train_model(
            train_csv=train_csv,
            data_root=data_root,
            out_dir=out_dir,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            val_ratio=val_ratio,
            seed=seed,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            num_workers=num_workers,
            use_amp=use_amp,
            sample_frac=sample_frac,
            max_train=max_train,
            max_val=max_val,
            image_size=image_size,
            num_slices=num_slices,
            enable_fast_read=enable_fast_read,
            max_steps_per_epoch=max_steps_per_epoch,
        )

    use_local_checkpoint(model_name, best_path)
    return best_path


    

# ============== Main Execution ==============
if __name__ == "__main__":
    """
    参照竞赛 notebook 的执行入口：
    - train 子命令：进行训练并保存 best.pth（可 --serve-after 训练后立刻启动服务）
    - serve 子命令：仅启动服务（Kaggle/本地网关）
    - predict 子命令：本地对单个序列目录推理
    - 若直接运行且找不到权重，且设置了 KAGGLE_AUTO_TRAIN=1，则自动训练后再启动服务
    """
    parser = argparse.ArgumentParser(description="RSNA IAD inference/train entry")
    sub = parser.add_subparsers(dest="cmd")

    # train 子命令
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--train-csv", type=str, default=None, help="train.csv 路径或数据集根目录")
    p_train.add_argument("--data-root", type=str, default=None, help="series 目录或数据集根目录")
    p_train.add_argument("--out-dir", type=str, default="/kaggle/working/out", help="输出目录")
    p_train.add_argument("--model-name", type=str, default="tf_efficientnetv2_s")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-5)
    p_train.add_argument("--val-ratio", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--pretrained-backbone", action="store_true", default=True)
    p_train.add_argument("--no-pretrained-backbone", dest="pretrained-backbone", action="store_false")
    p_train.add_argument("--freeze-backbone", action="store_true", default=True)
    p_train.add_argument("--no-freeze-backbone", dest="freeze-backbone", action="store_false")
    p_train.add_argument("--unfreeze-last-n", type=int, default=0, help="冻结骨干时仍解冻最后 N 个顶层子模块参与训练")
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--no-amp", dest="use_amp", action="store_false", default=True)
    p_train.add_argument("--serve-after", action="store_true", help="训练完成后立刻启动服务")
    # Fast-tune args
    p_train.add_argument("--sample-frac", type=float, default=0.5, help="按比例抽样训练数据 (0-1]，默认 0.5；验证集默认不抽样")
    p_train.add_argument("--max-train", type=int, default=None, help="训练集最多样本数（与 sample-frac 二选一）")
    p_train.add_argument("--max-val", type=int, default=None, help="验证集最多样本数")
    p_train.add_argument("--ft-image-size", type=int, default=None, help="覆盖输入分辨率以加速，如 384/320")
    p_train.add_argument("--ft-num-slices", type=int, default=None, help="覆盖切片数以加速，如 16")
    p_train.add_argument("--enable-fast-read", action="store_true", help="仅抽样读取 DICOM 文件，减少 I/O")
    p_train.add_argument("--max-steps-per-epoch", type=int, default=None, help="每轮最多训练多少步，用于快速验证")

    # serve 子命令（默认）：启动前先训练（若存在已训练的 best.pth 则复用）
    p_serve = sub.add_parser("serve", help="Train-then-start inference server (default)")
    p_serve.add_argument("--train-csv", type=str, default=None, help="train.csv 路径或数据集根目录")
    p_serve.add_argument("--data-root", type=str, default=None, help="series 目录或数据集根目录")
    p_serve.add_argument("--out-dir", type=str, default="/kaggle/working/out", help="输出目录")
    p_serve.add_argument("--model-name", type=str, default="tf_efficientnetv2_s")
    p_serve.add_argument("--epochs", type=int, default=5)
    p_serve.add_argument("--batch-size", type=int, default=4)
    p_serve.add_argument("--lr", type=float, default=3e-4)
    p_serve.add_argument("--weight-decay", type=float, default=1e-5)
    p_serve.add_argument("--val-ratio", type=float, default=0.2)
    p_serve.add_argument("--seed", type=int, default=42)
    p_serve.add_argument("--pretrained-backbone", action="store_true", default=True)
    p_serve.add_argument("--no-pretrained-backbone", dest="pretrained-backbone", action="store_false")
    p_serve.add_argument("--freeze-backbone", action="store_true", default=True)
    p_serve.add_argument("--no-freeze-backbone", dest="freeze-backbone", action="store_false")
    p_serve.add_argument("--unfreeze-last-n", type=int, default=2, help="冻结骨干时仍解冻最后 N 个顶层子模块参与训练")
    p_serve.add_argument("--num-workers", type=int, default=2)
    p_serve.add_argument("--no-amp", dest="use_amp", action="store_false", default=True)
    p_serve.add_argument("--no-reuse", dest="reuse_if_exists", action="store_false", default=True,
                         help="Do not reuse existing best.pth; force retraining (toggle to control reuse vs retrain)")
    # Fast-tune args
    p_serve.add_argument("--sample-frac", type=float, default=0.5, help="按比例抽样训练数据 (0-1]，默认 0.5；验证集默认不抽样")
    p_serve.add_argument("--max-train", type=int, default=None, help="训练集最多样本数（与 sample-frac 二选一）")
    p_serve.add_argument("--max-val", type=int, default=None, help="验证集最多样本数")
    p_serve.add_argument("--ft-image-size", type=int, default=None, help="覆盖输入分辨率以加速，如 384/320")
    p_serve.add_argument("--ft-num-slices", type=int, default=None, help="覆盖切片数以加速，如 16")
    p_serve.add_argument("--enable-fast-read", action="store_true", help="仅抽样读取 DICOM 文件，减少 I/O")
    p_serve.add_argument("--max-steps-per-epoch", type=int, default=None, help="每轮最多训练多少步，用于快速验证")
    

    # predict 子命令（本地单序列）
    p_pred = sub.add_parser("predict", help="Run local single-series prediction")
    p_pred.add_argument("series_path", type=str, help="单个序列目录路径")

    

    # 无子命令时默认走 serve
    parser.set_defaults(cmd="serve")
    
    def _running_in_notebook() -> bool:
        try:
            # 在 Kaggle/Colab/Notebook 环境下都会存在 IPython/ipykernel
            import IPython  # noqa: F401
            return True
        except Exception:
            return False
    
    # 在 Notebook 环境中忽略内核注入的 -f/JSON 等参数，使用默认命令
    in_nb = _running_in_notebook()
    if in_nb:
        args, _unknown = parser.parse_known_args([])
    else:
        args, _unknown = parser.parse_known_args()

    # 若未显式指定子命令，且未注入对应子命令参数，补一次解析以附加子命令默认参数
    if args.cmd in {"serve", "train"} and not hasattr(args, "train_csv"):
        args, _unknown = parser.parse_known_args([args.cmd])

    def _start_server():
        if rsna_server is not None:
            inference_server = rsna_server.RSNAInferenceServer(predict)
            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                inference_server.serve()
            else:
                inference_server.run_local_gateway()
                sub_path = "/kaggle/working/submission.parquet"
                if os.path.exists(sub_path):
                    try:
                        submission_df = pl.read_parquet(sub_path)
                        display(submission_df)
                    except Exception as e:
                        print(f"[Warn] Unable to display parquet: {e}")
        else:
            print("[Info] kaggle_evaluation not available. For local test, use: python effi.py predict <series_path>")

    if args.cmd == "train":
        best = train_model(
            train_csv=getattr(args, "train_csv", None),
            data_root=getattr(args, "data_root", None),
            out_dir=args.out_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            seed=args.seed,
            pretrained_backbone=getattr(args, "pretrained_backbone", True),
            freeze_backbone=getattr(args, "freeze_backbone", True),
            unfreeze_last_n_layers=getattr(args, "unfreeze_last_n", 0),
            num_workers=args.num_workers,
            use_amp=args.use_amp,
            sample_frac=getattr(args, "sample_frac", None),
            max_train=getattr(args, "max_train", None),
            max_val=getattr(args, "max_val", None),
            image_size=getattr(args, "ft_image_size", None),
            num_slices=getattr(args, "ft_num_slices", None),
            enable_fast_read=getattr(args, "enable_fast_read", False),
            max_steps_per_epoch=getattr(args, "max_steps_per_epoch", None),
        )
        use_local_checkpoint(args.model_name, best)
        if args.serve_after:
            try:
                load_models()
            except Exception as e:
                print(f"[Warn] load_models failed after training: {e}")
            _start_server()
        sys.exit(0)

    if args.cmd == "predict":
        series_path = args.series_path
        if not os.path.isdir(series_path):
            print(f"[Error] series_path not found: {series_path}")
            sys.exit(1)
        try:
            df = predict(series_path)
            print(df)
        except Exception as e:
            print(f"[Error] Local prediction failed: {e}")
        sys.exit(0)

    

    if args.cmd == "serve":
        # 先训练（或复用已训练权重），再注册并加载，然后启动评测服务
        try:
            best = prepare_trained_model_for_serving(
                train_csv=getattr(args, "train_csv", None),
                data_root=getattr(args, "data_root", None),
                out_dir=args.out_dir,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                val_ratio=args.val_ratio,
                seed=args.seed,
                pretrained_backbone=getattr(args, "pretrained_backbone", True),
                freeze_backbone=getattr(args, "freeze_backbone", True),
                unfreeze_last_n_layers=getattr(args, "unfreeze_last_n", 0),
                num_workers=args.num_workers,
                use_amp=args.use_amp,
                reuse_if_exists=args.reuse_if_exists,
                sample_frac=getattr(args, "sample_frac", None),
                max_train=getattr(args, "max_train", None),
                max_val=getattr(args, "max_val", None),
                image_size=getattr(args, "ft_image_size", None),
                num_slices=getattr(args, "ft_num_slices", None),
                enable_fast_read=getattr(args, "enable_fast_read", False),
                max_steps_per_epoch=getattr(args, "max_steps_per_epoch", None),
            )
            print(f"[Serve] Using checkpoint: {best}")
        except Exception as e:
            print(f"[Error] prepare_trained_model_for_serving failed: {e}")
            sys.exit(1)

        try:
            load_models()
        except Exception as e:
            print(f"[Warn] load_models failed after training: {e}")

        _start_server()
