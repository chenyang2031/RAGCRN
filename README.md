# RAGCRN: Reinforced Attentive Graph Convolution Recurrent Network

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![views since 2025/02/21](https://komarev.com/ghpvc/?username=muying&color=5865f2)

RAGCRN is a deep learning model based on Graph Convolutional Networks (GCN) and Gated Recurrent Units (GRU), designed for spatiotemporal data prediction. This model incorporates attention mechanisms, enabling it to effectively capture dynamic dependencies in spatiotemporal data, making it suitable for traffic flow prediction, weather forecasting, and other tasks.

## Project Structure

```
RAGCRN/
├── database/
│   └── PEMS08/                # PEMS08 dataset
├── PEMS08_adj.csv             # PEMS08 adjacency matrix file
├── RAGCRN.py                  # RAGCRN model implementation
├── data_preprocess.py         # Data preprocessing script
└── README.md                  # Project documentation
```

## Key Features

- **Graph Convolutional Network (GCN)**: Captures spatial dependencies.
- **Gated Recurrent Unit (GRU)**: Captures temporal dependencies.
- **Attention Mechanism**: Dynamically adjusts node importance.
- **Modular Design**: Easy to extend and modify.

## Quick Start

### Environment Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Matplotlib (for visualization)

### Installing Dependencies

```bash
pip install torch numpy pandas matplotlib
```

### Data Preparation

1. Download the [PEMS08](https://github.com/chenyang2031/RAGCRN/tree/main/database/PEMS08) dataset.
2. Place the dataset in the `database/PEMS08/` directory.
3. Ensure the `PEMS08_adj.csv` file is located in the project root directory.

### Running an Example

```python
from RAGCRN import RAGCRN, Config

# Initialize configuration
config = Config()

# Initialize model
model = RAGCRN(config).to(device)

# Generate random input data
batch_size = 64
seq_len = 12
node = config.node
train_data = torch.randn(batch_size, node, seq_len).to(device)
target_data = torch.randn(batch_size, node, config.n_pred).to(device)

# Run the model
output = model(train_data, target_data, 0)
print("Output shape:", output.shape)
```

### Visualizing Results

```python
from RAGCRN import visualize_predictions

# Visualize predictions for the first node
node_idx = 0
time_steps = [0, 1, 2]
visualize_predictions(output, node_idx, time_steps, ground_truth=target_data)
```

## Code Structure

- **`RAGCRN.py`**: Contains the core model implementation, including `GCNGRUCell`, `EndGCNGRU`, `GCN`, `Attention`, `Decoder` and other modules.
- **`data_preprocess.py`**: Data preprocessing script for loading and preprocessing the PEMS08 dataset.
- **`PEMS08_adj.csv`**: Adjacency matrix file for the PEMS08 dataset.

## Contributing

Contributions, issue reports, and improvement suggestions are welcome! Please follow these steps:

1. Fork this project.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add some feature'`).
4. Push the branch (`git push origin feature/YourFeature`).
5. Submit a Pull Request.

## License

This project is open source under the [MIT License](LICENSE).

---

## Tags

- **Deep Learning**: Graph Convolutional Network, Gated Recurrent Unit, Attention Mechanism.
- **Spatiotemporal Prediction**: Traffic Flow Prediction, Weather Forecasting.
- **PyTorch**: Implemented in PyTorch.
- **Open Source**: MIT License, contributions welcome.
