# A Geometric Algebra–Enhanced Spatio-temporal Learning Model for Reconstructing Sea Level Variability

## Repository Overview

This repository provides the experimental codebase for **sea-level variability reconstruction**, aiming to conduct systematic analysis and comparison of different model architectures and ablation settings.

---

## Repository Organization

The codebase is organized by **geographic regions**, where each region corresponds to an independent experimental configuration and dataset setup. The repository currently includes experiments for the following three regions:

* **North Atlantic – HIERRO**
  Experimental code for the North Atlantic HIERRO region, including ablation studies of the GA-enhanced model (Remove Linear Layer, Remove One Path, Remove Series Embedding, and Replace GAConvGRU with ConvGRU), as well as the main model (GA-enhanced model) and baseline models (SOFTS, GRU / LSTM / Transformer).

* **Northeast Pacific – SANTA MONICA**
  Experimental implementation for the Northeast Pacific SANTA MONICA region, featuring the same ablation settings as the HIERRO region, along with complete training and evaluation pipelines for the GA-enhanced model and multiple baseline models (SOFTS, GRU / LSTM / Transformer).

* **Southwest South China Sea – TANJUNG GELANG**
  Experimental code for the Southwest South China Sea TANJUNG GELANG region, covering identical model structures, ablation strategies, and baseline model configurations, enabling cross-region comparative analysis.

Each regional directory contains a **complete and independently executable** experimental code structure.

---

## Documentation Structure

To improve code readability and reproducibility, the repository adopts a **hierarchical documentation design**:

* **Top-level README**
  Provides an overview of the research background, overall code organization, and the relationships among different experimental regions.

* **Regional README files**
  Each regional directory includes its own README file, which details the experimental objectives, model configurations, data usage, and execution procedures specific to that region.

---

## Models and Experimental Settings

Across different regions, the repository implements and evaluates multiple time series forecasting models, including:

* **GA-enhanced model** (the proposed method)
* Classical baseline models (GRU, LSTM, Transformer, SOFTS)
* Ablation studies targeting key model components and structures

Detailed model implementations and experimental configurations are provided in the README files within the corresponding regional directories.

---

## Usage Guidance

Users are encouraged to select the regional directory of interest according to their research objectives and follow the instructions provided in the corresponding README file to run experiments and reproduce results.
The top-level README serves as an entry point for understanding the overall experimental design and repository organization.

---
