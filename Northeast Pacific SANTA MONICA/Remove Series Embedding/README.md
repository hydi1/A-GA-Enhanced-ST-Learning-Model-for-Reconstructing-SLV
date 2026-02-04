# GAConvGRU Long-Term Time Series Forecasting Framework

## Project Overview

This repository is intended for experimental research on long-term time series forecasting.
Based on a general long-sequence forecasting framework, it implements and extends the training
and evaluation pipeline of the **GA-enhanced ConvGRU (GAConvGRU)** model.

The codebase is designed for research and experimental purposes, supporting multi-random-seed
training, model evaluation, and component ablation studies, enabling systematic comparison
of different model structures and configurations within a unified framework.

The repository currently includes:
- Training and evaluation pipeline for the GAConvGRU model
- A reusable experimental framework for long-term time series forecasting

---

## Main Entry Points

- `GAConvGRU_run.py`  
  Main entry script for single- and multi-random-seed training and evaluation.

- `exp/exp_long_term_forecasting.py`  
  Core implementation of the long-term forecasting experiment pipeline,
  including training, validation, and testing stages.

---

## Repository Structure

```

SOFTS-main/
├── GAConvGRU_run.py        # Main training and evaluation entry
├── checkpoints/            # Directory for saving trained models
├── core_qnn/              # Quantum / kernel-related modules
├── data_provider/         # Data loading and preprocessing
├── exp/                   # Experiment pipelines
├── layers/                # Model layer definitions
├── models/                # Model architectures
├── test_result/           # Test results and outputs
├── utils/                 # Utility functions
└── README.md

````

---

## Environment Requirements

- Python 3.8 or later (recommended: 3.9 / 3.10)
- PyTorch (recommended: 1.12+ or 2.x)
- Required Python packages (based on source imports):
  - `numpy`
  - `pandas`
  - `torchinfo`

---

## Data Preparation

1. Prepare input data files. Example configurations can be found in
   `GAConvGRU_run.py` (parameters: `root_path`, `data_path`, `target_path`).
2. Ensure that:
   - `data_path` points to input data in `.npy` format
   - `target_path` points to the target `.xlsx` file (consistent with the code example)
3. Modify the corresponding paths in `GAConvGRU_run.py` to match your local data setup.

---

## Quick Start

Run the following command under the project root directory:

```bash
python GAConvGRU_run.py
````

This script will:

* Train the model using a predefined list of random seeds
* Report evaluation metrics for each random seed
* Automatically select and display results from the best-performing seed

---

---

## Common Configuration Options (`GAConvGRU_run.py`)

* `seq_len` / `label_len` / `pred_len`
  Sequence length settings for input, label, and prediction horizons.

* `input_size` / `hidden_size` / `output_size`
  Input, hidden, and output dimensions of the model.

* `use_gpu`, `device_ids`
  GPU and multi-GPU configuration.

* `checkpoints`
  Path for saving model checkpoints (default: located under the project root directory as `checkpoints/`).

---

## Output Description

* Training logs are printed to the console
* Model parameters and intermediate results are saved under the `checkpoints/` directory
* Evaluation results from the testing stage are stored in the `test_result/` directory
