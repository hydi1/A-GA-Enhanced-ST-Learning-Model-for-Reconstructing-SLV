Below is a clean **English translation** of your README content, keeping the original structure and technical meaning intact.

---

# SOFTS Sea Level Variability Prediction and Reconstruction Experimental Framework

## Project Overview

This repository is designed for experimental research on **sea level variability prediction and reconstruction**.
It is based on the **SOFTS (State-of-the-art long-term Time Series forecasting)** model and implements a complete training and evaluation pipeline for **multivariate time series forecasting**.

The codebase is intended for **scientific research and experimental use**, providing a unified workflow for data preprocessing, training, validation, and testing.
It can be used as a **baseline model or a comparative method** for sea level variability prediction tasks.

---

## Main Entry Points

* `softsrun.py`
  Main script for training and evaluation, including model configuration and random seed settings.

* `models/SOFTS.py`
  Implementation of the SOFTS model.

---

## Directory Structure

```
.
├── README.md
├── output_test/                # Output artifacts (generated)
└── SOFTS-main/
    ├── softsrun.py             # Main entry for training and evaluation
    ├── Reconstruction.py       # Reconstruction utilities / script
    ├── scaler_x_time.pkl       # Saved scaler (generated)
    ├── scaler_y_time.pkl       # Saved scaler (generated)
    ├── checkpoints/            # Model checkpoint output directory
    ├── data_provider/          # Data loading and DataLoader definitions
    ├── exp/                    # Experiment pipelines (train / validation / test)
    ├── layers/                 # Network layers and embedding modules
    ├── models/                 # SOFTS model definitions
    ├── test_result/            # Evaluation outputs (generated)
    └── utils/                  # Utility functions and time feature encodings
```

---

## Environment and Dependencies

* Python 3.8 or higher
* PyTorch (recommended to match your CUDA/GPU environment)
* Main dependencies (based on current imports):

  * `torch`
  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `torchinfo`
  * `joblib`

Installation example:

```bash
pip install torch numpy pandas scikit-learn torchinfo joblib
```

If GPU acceleration is required, please ensure that PyTorch is installed with a CUDA version compatible with your system.

---

## Data Format and Preprocessing

The current data loading logic is implemented in `data_provider/data_loader.py`, with the following default conventions:

### Input Features

* Stored as `.npy` files under `root_path/data_path`
* After loading, the data is reshaped as:

  ```
  reshape(-1, 4 * 29 * 6 * 11)
  ```

  to form the model input features.

### Target Values

* Stored as `.xlsx` files under `target_path`
* By default:

  * The first column represents the time index
  * (Other columns correspond to target variables)

### Dataset Splitting

* The dataset is split **chronologically** into training, validation, and test sets
* Typical ratio: **60% / 30% / 10%**

### Normalization

* When `scale=True`, the following files are saved to the running directory:

  * `scaler_x_time.pkl`
  * `scaler_y_time.pkl`

These files are used for inverse normalization and result reconstruction.

> If your data dimensions, number of variables, or column structure differ from the above assumptions,
> please modify the data loading and reshape logic in `data_provider/data_loader.py` accordingly.

---

## Quick Start

1. Modify the parameters in `softsrun.py` according to your local environment and data paths, especially:

   * `root_path`
   * `data_path`
   * `target_path`
   * `checkpoints`

2. Run training and evaluation from the project root directory:

```bash
python softsrun.py
```

---

## Training and Evaluation Outputs

* Console logs report training, validation, and test losses for each epoch
* Evaluation metrics include **RMSE** and **MAE**
* Model checkpoints are saved in the directory specified by `checkpoints/`

---

## Frequently Asked Questions

* **Path issues**
  Ensure that all paths in the scripts match your local file structure. Avoid using invalid absolute paths.

* **Data dimension mismatch**
  Verify that the actual dimensions of the `.npy` files are consistent with the reshape logic defined in the code.

---
