import os
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .config import CFG


def _norm_model_key(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name).upper()


def discover_timm_backbone_checkpoint(model_name: str) -> Optional[str]:
    env_direct = os.environ.get("TIMM_CHECKPOINT")
    if env_direct and os.path.isfile(env_direct):
        return env_direct
    key = f"TIMM_CHECKPOINT_{_norm_model_key(model_name)}"
    env_model = os.environ.get(key)
    if env_model and os.path.isfile(env_model):
        return env_model

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
    def __init__(self, model_name: str, num_classes: int = 14, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2, freeze_backbone: bool = False,
                 backbone_checkpoint_path: Optional[str] = None, unfreeze_last_n_layers: int = 0,
                 use_meta_features: bool = False) -> None:
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.use_meta_features = bool(use_meta_features)

        if 'swin' in model_name:
            self.backbone = timm.create_model(
                model_name,
                pretrained=(pretrained and backbone_checkpoint_path is None),
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,
                num_classes=0,
                global_pool='' 
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=(pretrained and backbone_checkpoint_path is None),
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,
                global_pool=''
            )

        if backbone_checkpoint_path and os.path.isfile(backbone_checkpoint_path):
            try:
                ckpt = torch.load(backbone_checkpoint_path, map_location='cpu')
                state = None
                if isinstance(ckpt, dict):
                    for k in ['state_dict', 'model', 'weights', 'params']:
                        v = ckpt.get(k)
                        if isinstance(v, dict):
                            state = v
                            break
                    if state is None:
                        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)} or ckpt
                else:
                    state = ckpt
                self.backbone.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[Offline] Failed to load backbone checkpoint: {e}")

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                num_features = features.shape[1]
                self.needs_pool = True
                self.needs_seq_pool = False
            elif len(features.shape) == 3:
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False

        if getattr(self, 'needs_pool', False):
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        meta_dim = 32 if self.use_meta_features else 0

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
        if getattr(self, 'freeze_backbone', False):
            with torch.no_grad():
                img_features = self.backbone(image)
        else:
            img_features = self.backbone(image)

        if getattr(self, 'needs_pool', False):
            img_features = self.global_pool(img_features).flatten(1)
        elif len(img_features.shape) == 4:
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            img_features = img_features.mean(dim=1)

        if self.use_meta_features and meta is not None:
            meta_features = self.meta_fc(meta)
            combined = torch.cat([img_features, meta_features], dim=1)
        else:
            combined = img_features

        return self.classifier(combined)
