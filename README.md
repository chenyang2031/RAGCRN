---

# RAGCRN: Recurrent Attention Graph Convolutional Recurrent Network

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

RAGCRN 是一个基于图卷积网络（GCN）和循环神经网络（GRU）的深度学习模型，用于时空数据预测。该模型结合了注意力机制，能够有效捕捉时空数据中的动态依赖关系，适用于交通流量预测、气象预测等任务。

## 项目结构

```
RAGCRN/
├── database/
│   └── PEMS08/                # PEMS08 数据集
├── PEMS08_adj.csv             # PEMS08 邻接矩阵文件
├── RAGCRN.py                  # RAGCRN 模型实现
├── data_preprocess.py         # 数据预处理脚本
└── README.md                  # 项目说明文档
```

## 主要特性

- **图卷积网络（GCN）**：用于捕捉空间依赖关系。
- **循环神经网络（GRU）**：用于捕捉时间依赖关系。
- **注意力机制**：动态调整节点的重要性。
- **模块化设计**：易于扩展和修改。

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Matplotlib（用于可视化）

### 安装依赖

```bash
pip install torch numpy pandas matplotlib
```

### 数据准备

1. 下载 [PEMS08](https://github.com/chenyang2031/RAGCRN/tree/main/database/PEMS08) 数据集。
2. 将数据集放置在 `database/PEMS08/` 目录下。
3. 确保 `PEMS08_adj.csv` 文件位于项目根目录。

### 运行示例

```python
from RAGCRN import RAGCRN, Config

# 初始化配置
config = Config()

# 初始化模型
model = RAGCRN(config).to(device)

# 生成随机输入数据
batch_size = 64
seq_len = 12
node = config.node
train_data = torch.randn(batch_size, node, seq_len).to(device)
target_data = torch.randn(batch_size, node, config.n_pred).to(device)

# 运行模型
output = model(train_data, target_data, 0)
print("Output shape:", output.shape)
```

### 可视化结果

```python
from RAGCRN import visualize_predictions

# 可视化第一个节点的预测结果
node_idx = 0
time_steps = [0, 1, 2]
visualize_predictions(output, node_idx, time_steps, ground_truth=target_data)
```

## 代码结构

- **`RAGCRN.py`**：包含模型的核心实现，包括 `GCNGRUCell`、`EndGCNGRU`、`GCN`、`Attention`、`Decoder` 等模块。
- **`data_preprocess.py`**：数据预处理脚本，用于加载和预处理 PEMS08 数据集。
- **`PEMS08_adj.csv`**：PEMS08 数据集的邻接矩阵文件。

## 贡献

欢迎贡献代码、提出问题或提出改进建议！请遵循以下步骤：

1. Fork 本项目。
2. 创建新的分支（`git checkout -b feature/YourFeature`）。
3. 提交更改（`git commit -m 'Add some feature'`）。
4. 推送分支（`git push origin feature/YourFeature`）。
5. 提交 Pull Request。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 标签

- **深度学习**：图卷积网络、循环神经网络、注意力机制。
- **时空预测**：交通流量预测、气象预测。
- **PyTorch**：基于 PyTorch 实现。
- **开源**：MIT 许可证，欢迎贡献。

---


