import os

# Competition constants
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

SHARED_DIR = '/kaggle/shared'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SELECTED_MODEL = 'ensemble'

MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/best/pytorch/default/1/tf_efficientnetv2_s_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}


class InferenceConfig:
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    image_size = 512
    num_slices = 32
    use_windowing = True
    batch_size = 1
    use_amp = True
    use_tta = True
    tta_transforms = 4
    ensemble_weights = {
        'tf_efficientnetv2_s': 1,
        'convnext_small': 0,
        'swin_small_patch4_window7_224': 0
    }


CFG = InferenceConfig()
