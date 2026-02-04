# SOFTS 海平面变率预测与重建实验框架

## 项目简介

本仓库用于海平面变率预测与重建的实验研究，
基于 **SOFTS（State-of-the-art long-term time Series forecasting）** 模型，
实现了多变量时间序列的训练与评估流程。

该代码仓库面向科研与实验用途，支持统一的数据处理、训练、验证与测试流程，
可作为海平面变率预测任务中的基线模型或对比方法使用。

---

## 主要入口

- `softsrun.py`  
  训练与评估主入口脚本，包含模型参数配置与随机种子设置。

- `models/SOFTS.py`  
  SOFTS 模型的具体实现。

---

## 目录结构



.
├── softsrun.py # 训练与评估主入口
├── models/ # SOFTS 模型定义
├── layers/ # 网络层与嵌入模块
├── exp/ # 实验流程（训练 / 验证 / 测试）
├── data_provider/ # 数据读取与 DataLoader
├── checkpoints/ # 模型检查点输出目录
├── utils/ # 工具函数与时间特征
└── README.md


---

## 环境与依赖

- Python 3.8 及以上
- PyTorch（建议与 CUDA/GPU 环境匹配）
- 主要依赖库（根据当前代码导入）：
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `torchinfo`
  - `joblib`

安装示例：

```bash
pip install torch numpy pandas scikit-learn torchinfo joblib
```

如需使用 GPU，请确保已安装与 CUDA 版本匹配的 PyTorch。

## 数据格式与处理说明

当前数据读取逻辑位于 `data_provider/data_loader.py`，默认约定如下：

- 输入特征：位于 `root_path/data_path` 的 `.npy` 文件，数据在读取后通过 `reshape(-1, 4 * 29 * 6 * 11)` 转换为模型输入特征。
pip install torch numpy pandas scikit-learn torchinfo joblib


如需使用 GPU，请确保已安装与 CUDA 版本匹配的 PyTorch。

### 数据格式与处理说明

当前数据读取逻辑位于 data_provider/data_loader.py，默认约定如下：

## 数据格式与处理说明

当前数据读取逻辑位于 `data_provider/data_loader.py`，默认约定如下：

* **输入特征**
  位于 `root_path/data_path` 的 `.npy` 文件，
  数据在读取后通过
  `reshape(-1, 4 * 29 * 6 * 11)`
  转换为模型输入特征。

* **目标值**
  位于 `target_path` 的 `.xlsx` 文件，
  默认：

  * 第一列为时间索引
  *

* **数据划分方式**
  按时间顺序划分训练 / 验证 / 测试集，
  比例约为 **60% / 30% / 10%**。

* **归一化处理**
  当 `scale=True` 时，将保存：

  * `scaler_x_time.pkl`
  * `scaler_y_time.pkl`
    至运行目录，用于反归一化与结果重建。

> 若你的数据维度、变量数量或列结构与上述设定不同，
> 请相应修改 `data_provider/data_loader.py` 中的数据读取与 reshape 逻辑。

---

## 快速开始

1. 根据本地环境与数据路径，修改 `softsrun.py` 中的参数，尤其是：

   * `root_path`
   * `data_path`
   * `target_path`
   * `checkpoints`
2. 在项目根目录下运行训练与评估：

```bash
python softsrun.py
```

---

## 训练与评估输出

* 控制台日志将输出每个 epoch 的训练、验证与测试损失
* 评估阶段输出 **RMSE** 与 **MAE** 指标
* 模型检查点保存在 `checkpoints/` 指定目录下

---

## 常见问题

* **路径问题**
  请确认脚本中的路径配置与本地文件结构一致，避免使用失效的绝对路径。

* **数据尺寸不匹配**
  请检查 `.npy` 数据的实际维度是否与代码中的 reshape 逻辑一致。
