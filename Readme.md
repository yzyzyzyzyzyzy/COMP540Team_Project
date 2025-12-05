# RSNA Intracranial Aneurysm Detection - Final Project Codebase

## Project Overview
This repository contains the complete deep learning pipeline for our 2.5D Dual-Stream Siamese Network.
**For Grading:** We recommend starting with **`03_Inference.ipynb`**, which is configured to run out-of-the-box using provided pre-trained weights and sample data.

---

##  📂 Guide to Files
Here is the organization of the submitted codebase:

* **`01_Data_Preprocessing.ipynb`**
    * **Function:** Converts raw DICOM Series into 2.5D tensor slices (`.npy` format) to optimize I/O performance.
    * **Input:** Raw RSNA Dataset.
    * **Output:** Pre-processed tensors in a cache directory.
* **`02_Model_Training.ipynb`**
    * **Function:** Defines the `HybridAneurysmModel` (Siamese UNet Encoder) and executes the 5-fold cross-validation training loop.
    * **Output:** Model weights (`.pth`) saved to the disk.
* **`03_Inference.ipynb` (Run this for Grading)**
    * **Function:** Loads the pre-trained model, performs inference, and generates the Attention Map visualizations/Error Analysis shown in our report.
* **`models/`**: Contains our trained model weights
* **`rsna-cta-training.ipynb` (Run this for Grading)**
    * **Function:** This file is the same with the model training file, but include the implicit neuron representation model.


##  How to Download Data (Automated via KaggleHub)

We utilize the `kagglehub` library to automatically download both the raw competition data and the pre-processed tensor data directly within the notebook.

**Prerequisite:**
You will need to install the library first

```bash
pip install kagglehub
```

# How to run the code

Directly press run all.



In the 03_Inverence.ipynb, you can upload the provided `models/` folder. Load our pre-trained model weights manually. 

