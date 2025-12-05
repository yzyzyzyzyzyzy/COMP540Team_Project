import os
import pandas as pd


def _seed_everything(seed: int = 42):
    import random, numpy as _np, torch
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_kaggle_training_paths(train_csv: str | None, data_root: str | None):
    def is_series_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.basename(os.path.normpath(p)).lower() == 'series'

    if train_csv and os.path.isdir(train_csv):
        candidate = os.path.join(train_csv, 'train.csv')
        if os.path.isfile(candidate):
            train_csv = candidate

    if data_root and os.path.isdir(data_root) and not is_series_dir(data_root):
        series_candidate = os.path.join(data_root, 'series')
        if os.path.isdir(series_candidate):
            data_root = series_candidate

    if (not data_root or not os.path.isdir(data_root)) and train_csv and os.path.isfile(train_csv):
        root = os.path.dirname(os.path.abspath(train_csv))
        series_candidate = os.path.join(root, 'series')
        if os.path.isdir(series_candidate):
            data_root = series_candidate

    if (not train_csv or not os.path.isfile(train_csv)) and data_root and os.path.isdir(data_root):
        root = os.path.abspath(data_root)
        if is_series_dir(root):
            root = os.path.dirname(root)
        csv_candidate = os.path.join(root, 'train.csv')
        if os.path.isfile(csv_candidate):
            train_csv = csv_candidate

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

    if not train_csv or not os.path.isfile(train_csv):
        raise FileNotFoundError(f"未找到 train.csv，请检查传入路径或 Kaggle 数据集挂载。train_csv={train_csv}")
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(f"未找到 series 目录，请检查传入路径或 Kaggle 数据集挂载。data_root={data_root}")

    print(f"[Paths] train_csv = {train_csv}")
    print(f"[Paths] series_root = {data_root}")
    return train_csv, data_root
