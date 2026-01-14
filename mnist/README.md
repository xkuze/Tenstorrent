# MNIST

Handwritten digit classification (0-9) using MLP architecture.

## Structure

```
mnist/
├── model.py          # MLP LightningModule
├── train.py          # Training with Optuna hyperparameter search
├── utils.py          # MNISTDataModule
├── inference_ttnn.py # Inference on TensTorrent hardware
├── weights/          # Model checkpoints
└── logs/             # TensorBoard and CSV logs
```

## Architecture

Multi-Layer Perceptron with configurable hidden layers.

```
Input (784)           ← 28x28 flattened
    │
    ▼
┌─────────────────┐
│  Linear + ReLU  │   ← Hidden layer 1 (e.g., 256)
│  + Dropout      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Linear + ReLU  │   ← Hidden layer 2 (e.g., 128)
│  + Dropout      │
└─────────────────┘
    │
    ▼
Output (10)           ← Digit classes 0-9
```

## Training

```bash
python -m mnist.train
```

**Optuna Hyperparameter Search:**
- Hidden layers: 1-3 layers
- Hidden sizes: [64, 128, 256, 512]
- Dropout: 0.0-0.5
- Learning rate: 1e-4 to 1e-2 (log scale)
- Batch size: [32, 64, 128]
- Optimizer: [Adam, SGD, RMSprop]

**Training Config:**
- Trials: 20 (5 epochs each)
- Final training: 30 epochs
- Early stopping: patience=10

## Inference

```bash
# Run on TensTorrent device
python -m mnist.inference_ttnn --device_id 3

# With custom parameters
python -m mnist.inference_ttnn --device_id 3 --batch_size 64 --num_samples 100
```

**TT-NN Implementation:**
- Input flattened to [batch, 784]
- Linear layers: `ttnn.matmul` + `ttnn.add`
- Activation: `ttnn.relu`
- All operations on TensTorrent hardware

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.76% |
| PCC (PyTorch vs TT-NN) | > 0.99 |

## Files

| File | Description |
|------|-------------|
| `model.py` | MLP class with training/validation/test steps |
| `train.py` | Optuna search + final model training |
| `utils.py` | MNISTDataModule (60k train, 10k test) |
| `inference_ttnn.py` | TT-NN inference with PCC comparison |

## Hyperparameters (Best)

```json
{
  "hidden_sizes": [256, 128],
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "optimizer": "Adam"
}
```
