import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset

from .config import CFG
from .transforms import get_inference_transform, get_train_transform


def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)


def get_windowing_params(modality: str):
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350),
        'MRA': (600, 1200),
        'MRI': (40, 80),
    }
    return windows.get(modality, (40, 80))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
        return (arr * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    series_path = Path(series_path)
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()

    fast_read = os.environ.get("FAST_READ", "0") == "1"
    if fast_read and len(all_filepaths) > CFG.num_slices:
        idx = np.linspace(0, len(all_filepaths) - 1, CFG.num_slices).astype(int)
        all_filepaths = [all_filepaths[i] for i in idx]

    if len(all_filepaths) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata

    slices = []
    metadata = {}
    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)
            if img.ndim == 3:
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    img = img[:, :, 0]

            if i == 0:
                metadata['modality'] = getattr(ds, 'Modality', 'CT')
                try:
                    age_str = getattr(ds, 'PatientAge', '050Y')
                    age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
                    metadata['age'] = min(age, 100)
                except:
                    metadata['age'] = 50
                try:
                    sex = getattr(ds, 'PatientSex', 'M')
                    metadata['sex'] = 1 if sex == 'M' else 0
                except:
                    metadata['sex'] = 0

            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            if CFG.use_windowing:
                window_center, window_width = get_windowing_params(metadata['modality'])
                img = apply_dicom_windowing(img, window_center, window_width)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            img = cv2.resize(img, (CFG.image_size, CFG.image_size))
            slices.append(img)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    if len(slices) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    else:
        volume = np.array(slices)
        if len(slices) > CFG.num_slices:
            indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
            volume = volume[indices]
        elif len(slices) < CFG.num_slices:
            pad_size = CFG.num_slices - len(slices)
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')

    return volume, metadata


def build_three_channel_image(volume: np.ndarray) -> np.ndarray:
    middle_slice = volume[CFG.num_slices // 2]
    mip = np.max(volume, axis=0)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    std_proj = _normalize_to_uint8(std_proj)
    return np.stack([middle_slice, mip, std_proj], axis=-1)


def build_meta_from_metadata(metadata: Dict) -> np.ndarray:
    age_normalized = float(metadata.get('age', 50)) / 100.0
    sex = float(metadata.get('sex', 0))
    return np.array([age_normalized, sex], dtype=np.float32)


class RSNAIADTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform or get_train_transform()

    def __len__(self):
        return len(self.df)

    def _make_image_and_meta(self, series_path: str):
        volume, metadata = process_dicom_series(series_path)
        image = build_three_channel_image(volume)
        age_normalized = float(metadata.get('age', 50)) / 100.0
        sex = float(metadata.get('sex', 0))
        meta = np.array([age_normalized, sex], dtype=np.float32)
        return image, meta

    def __getitem__(self, idx: int):
        from .config import ID_COL, LABEL_COLS
        row = self.df.iloc[idx]
        uid = str(row[ID_COL])
        series_path = os.path.join(self.root_dir, uid)
        image, meta = self._make_image_and_meta(series_path)
        image = self.transform(image=image)['image']
        target = torch.tensor(row[LABEL_COLS].astype(np.float32).values)
        meta_tensor = torch.from_numpy(meta)
        return image, meta_tensor, target
