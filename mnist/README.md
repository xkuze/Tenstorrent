# MNIST

Handwritten digit classification (0-9). MLP architecture trained with Optuna hyperparameter search.

## Structure

```
mnist/
├── model.py          # MLP definition
├── train.py          # Training with Optuna
├── utils.py          # DataModule
├── inference_ttnn.py # Inference on TensTorrent
├── weights/          # Checkpoints
└── logs/             # Training logs
```

## Training

```bash
python -m mnist.train
```

Optuna searches through hidden layer sizes, dropout rates, learning rates, and optimizers. Best config trains for 30 epochs with early stopping.

## Inference

```bash
python -m mnist.inference_ttnn --device_id 2
```

Runs on TensTorrent hardware. Compares PyTorch vs TT-NN outputs using PCC metric.

## Results

Test accuracy: 97.76%
