# RSNA IAD – EfficientNet Training and Inference 

This folder contains a simple, ready-to-run entry script plus a modular library:

- `kaggle_train.py`: one-stop script for training and inference.
  - Train a model and save checkpoints
  - Serve an inference 
- `train_lib/`: readable, maintainable components split by responsibility
  - `config.py`: configuration helpers
  - `data.py`: dataset and data loading utilities
  - `engine.py`: train/validate loops
  - `models.py`: model definitions
  - `transforms.py`: albumentations pipelines
  - `utils.py`: misc helpers

## Quick start

You can use `kaggle_train.py` directly without touching the submodules.

### Train

```powershell
# Example (PowerShell): train with defaults and write outputs to /kaggle/working/out
python .\train\kaggle_train.py train --train-csv <path-or-root> --data-root <series-dir-or-root> --out-dir <output_dir>
```



### Serve (train then start inference server)

```powershell
python .\train\kaggle_train.py serve --train-csv <path-or-root> --data-root <series-dir-or-root> --out-dir <output_dir>
```
This will train (or reuse an existing best checkpoint), register the model, load it, and start the inference server expected by Kaggle.

Tip: control whether to reuse an existing checkpoint or retrain from scratch using `--no-reuse`:

- Reuse if exists (default): do nothing if `<out_dir>/<model_name>_best.pth` already exists.
- Force retrain: add `--no-reuse` to ignore existing checkpoints and retrain.

Examples:

```powershell
# Reuse if a best checkpoint exists (default)
python .\train\kaggle_train.py serve --train-csv <path-or-root> --data-root <series-dir-or-root> --out-dir <output_dir>

# Force retrain even if a best checkpoint exists
python .\train\kaggle_train.py serve --train-csv <path-or-root> --data-root <series-dir-or-root> --out-dir <output_dir> --no-reuse
```

### Local predict (single series folder)

```powershell
python .\train\kaggle_train.py predict --series-path <path_to_series_folder>
```
Prints a row of probabilities for the required RSNA IAD labels.

## Notes


- The library (`train_lib/`) mirrors the same logic in smaller pieces for readability and easier maintenance.
- If you already have a trained checkpoint, you can register it and use it for inference.
- We provide our best model in the same directory.
