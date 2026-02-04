# GRU / LSTM / Transformer Long-Term Time Series Forecasting Framework

## Project Overview

This repository is intended for experimental research on sea level variability reconstruction.
Based on a long-term time series forecasting framework, it implements and evaluates multiple
classical sequence models, including **GRU, LSTM, and Transformer**.

The codebase is designed for research and experimental purposes, supporting multi-random-seed
training, model evaluation, and comparative analysis of results. It enables systematic
investigation of the forecasting performance of different model architectures within a unified
data processing and experimental pipeline.

The repository currently includes:
- Training and testing pipelines for GRU, LSTM, and Transformer models
- A reusable experimental framework for long-term time series forecasting

---

## Main Entry Points

- `main/GRUrun.py`  
  Main entry script for single- and multi-random-seed training and evaluation,
  supporting GRU, LSTM, and Transformer models.

- `main/exp/exp_long_term_forecasting.py`  
  Core implementation of the long-term time series forecasting experiment pipeline,
  including training, validation, and testing stages.

- `main/Reconstruction.py`  
  Scripts for reconstruction and post-processing of prediction results.

---

## Repository Structure

.
├── main/
│ ├── GRUrun.py # Main training and evaluation entry
│ ├── Reconstruction.py # Reconstruction / post-processing script
│ ├── checkpoints/ # Directory for saving model weights
│ ├── data_provider/ # Data loading and preprocessing
│ ├── exp/ # Experiment pipelines
│ ├── layers/ # Model building blocks
│ ├── models/ # Model definitions (GRU / LSTM / Transformer)
│ ├── test_result/ # Test result outputs
│ └── utils/ # Utility functions and time features
├── output_test/ # Test output files (examples or historical results)
├── scaler_x_time.pkl # Input data scaler (if used)
├── scaler_y_time.pkl # Output data scaler (if used)
└── README.md


---

## Environment Requirements

- Python 3.8 or later
- PyTorch (version should be compatible with the CUDA/GPU environment)
- Required dependencies (based on source imports):
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `torchinfo`

---

## Data Preparation

1. Prepare input data files. Example configurations can be found in
   `main/GRUrun.py` (parameters: `root_path`, `data_path`, `target_path`).
2. Ensure that:
   - `data_path` points to input data in `.npy` format
   - `target_path` points to the target `.xlsx` file (consistent with the code examples)
3. If data scalers are used, ensure that the paths to
   `scaler_x_time.pkl` and `scaler_y_time.pkl` are correctly specified.

---

## Quick Start

Run the following command under the project root directory:

```bash
python main/GRUrun.py
