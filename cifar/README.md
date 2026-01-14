# CIFAR-10

Image classification across 10 categories using CNN architecture.

## Structure

```
cifar/
├── model.py          # CNN LightningModule
├── train.py          # Training with Optuna hyperparameter search
├── utils.py          # CIFAR10DataModule
├── inference_ttnn.py # Inference on TensTorrent hardware
├── weights/          # Model checkpoints
└── logs/             # TensorBoard and CSV logs
```

## Categories

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Architecture

Convolutional Neural Network with configurable conv layers.

```
Input (3, 32, 32)      ← RGB image
    │
    ▼
┌──────────────────┐
│  Conv2d + ReLU   │   ← Conv layer 1 (e.g., 32 filters)
│  MaxPool + Drop  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Conv2d + ReLU   │   ← Conv layer 2 (e.g., 64 filters)
│  MaxPool + Drop  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Conv2d + ReLU   │   ← Conv layer 3 (e.g., 128 filters)
│  MaxPool + Drop  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Flatten         │
│  Linear + ReLU   │   ← FC layer (512 units)
│  Linear          │   ← Output (10 classes)
└──────────────────┘
```

## Training

```bash
python -m cifar.train
```

**Optuna Hyperparameter Search:**
- Conv layers: 2-4 layers
- Filters per layer: [32, 64, 128, 256]
- Kernel size: [3, 5]
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
python -m cifar.inference_ttnn --device_id 3

# With custom parameters
python -m cifar.inference_ttnn --device_id 3 --batch_size 32 --num_samples 64
```

**TT-NN Implementation (Hybrid):**
- Conv layers: Run on CPU (ttnn.conv2d requires special setup)
- FC layers: Run on TensTorrent (`ttnn.matmul` + `ttnn.add`)
- Activation: `ttnn.relu`

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 75.36% |
| PCC (PyTorch vs TT-NN) | > 0.99 |

## Files

| File | Description |
|------|-------------|
| `model.py` | CNN class with conv/fc layers |
| `train.py` | Optuna search + final model training |
| `utils.py` | CIFAR10DataModule (50k train, 10k test) |
| `inference_ttnn.py` | Hybrid inference (Conv CPU + FC TT-NN) |

## Hyperparameters (Best)

```json
{
  "num_filters": [32, 64, 128],
  "kernel_size": 3,
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "optimizer": "Adam"
}
```

## Notes

- Conv2d on TT-NN requires specific tensor layouts and padding
- Current implementation uses hybrid approach for simplicity
- Full TT-NN conv implementation would improve performance
