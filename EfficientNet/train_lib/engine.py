import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from .config import CFG
from .transforms import get_inference_transform, get_tta_transforms


def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, use_amp=True, log_interval=10):
    model.train()
    running_loss = 0.0
    count = 0
    last_end = time.time()
    max_steps_env = os.environ.get('MAX_STEPS_PER_EPOCH')
    max_steps = int(max_steps_env) if max_steps_env and max_steps_env.isdigit() else None

    device = next(model.parameters()).device
    for step, (images, metas, targets) in enumerate(loader):
        now = time.time()
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        comp_start = time.time()
        with autocast(enabled=use_amp):
            logits = model(images, metas)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        comp_time = time.time() - comp_start
        last_end = time.time()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

        if (step + 1) % log_interval == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(loader)} | Loss {loss.item():.4f} | comp {comp_time:.3f}s")

        if max_steps is not None and (step + 1) >= max_steps:
            print(f"[FastTune] Stop early at step {step+1} / {len(loader)} for epoch {epoch}.")
            break

    return running_loss / max(1, count)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    device = next(model.parameters()).device
    for images, metas, targets in loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images, metas)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    return total_loss / max(1, count)


def predict_single_model(model: torch.nn.Module, image: np.ndarray, meta_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    predictions = []
    TTA_TRANSFORMS = get_tta_transforms() if CFG.use_tta else None
    if CFG.use_tta and TTA_TRANSFORMS:
        for transform in TTA_TRANSFORMS[:CFG.tta_transforms]:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(enabled=CFG.use_amp):
                    output = model(aug_image, meta_tensor)
                    pred = torch.sigmoid(output)
                    predictions.append(pred.cpu().numpy())
        return np.mean(predictions, axis=0).squeeze()
    else:
        image_tensor = get_inference_transform()(image=image)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                output = model(image_tensor, meta_tensor)
                return torch.sigmoid(output).cpu().numpy().squeeze()


def predict_ensemble(models: dict, image: np.ndarray, meta_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    all_predictions = []
    weights = []
    for model_name, model in models.items():
        pred = predict_single_model(model, image, meta_tensor, device)
        all_predictions.append(pred)
        weights.append(CFG.ensemble_weights.get(model_name, 1.0))
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    return np.average(predictions, weights=weights, axis=0)
