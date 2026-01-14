# CIFAR-10

Image classification across 10 categories (airplane, car, bird, etc.). CNN architecture with Optuna tuning.

## Structure

```
cifar/
├── model.py          # CNN definition
├── train.py          # Training with Optuna
├── utils.py          # DataModule
├── inference_ttnn.py # Inference on TensTorrent
├── weights/          # Checkpoints
└── logs/             # Training logs
```

## Training

```bash
python -m cifar.train
```

Optuna explores conv layer counts, filter sizes, kernel dimensions, and dropout. Trains best model for 30 epochs.

## Inference

```bash
python -m cifar.inference_ttnn --device_id 2
```

Conv layers run on CPU, FC layers on TensTorrent device. Hybrid approach due to ttnn.conv2d complexity.

## Results

Test accuracy: 75.36%
