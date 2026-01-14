# TensTorrent ML Project

Training and inference on TensTorrent hardware. Three tasks: digit classification, image classification, and segmentation.

## Quick Start

```bash
# SSH to server (requires VPN)
ssh ekaterina_kuzmina1@10.30.0.207

# Activate environment
cd ~/tenstorrent
source .venv/bin/activate

# Run inference (replace N with device 0-7)
python -m mnist.inference_ttnn --device_id N
python -m cifar.inference_ttnn --device_id N
python -m unet.inference_ttnn --device_id N
```

## Project Structure

```
tenstorrent/
├── mnist/            # Digit classification (MLP)
│   ├── model.py
│   ├── train.py
│   ├── inference_ttnn.py
│   ├── weights/
│   └── logs/
├── cifar/            # Image classification (CNN)
│   ├── model.py
│   ├── train.py
│   ├── inference_ttnn.py
│   ├── weights/
│   └── logs/
├── unet/             # Segmentation (VGG19 encoder)
│   ├── model.py
│   ├── train.py
│   ├── inference_ttnn.py
│   ├── benchmark.py
│   ├── weights/
│   └── logs/
├── common/           # Shared utilities
│   └── metrics.py
├── configs/          # Configuration files
└── data/             # Datasets (gitignored)
```

## Training

All models use PyTorch Lightning. MNIST and CIFAR include Optuna hyperparameter search.

```bash
python -m mnist.train
python -m cifar.train
python -m unet.train
```

## Results

| Task | Architecture | Test Accuracy |
|------|--------------|---------------|
| MNIST | MLP | 97.76% |
| CIFAR-10 | CNN | 75.36% |
| UNet | VGG19 encoder | Dice ~0.29 |

## Tools

- **uv** — package manager
- **ruff** — linting and formatting
- **PyTorch Lightning** — training framework
- **Optuna** — hyperparameter optimization
- **TTNN** — TensTorrent inference library

## Device Selection

Eight devices available (0-3 local, 4-7 remote). Check Teams chat "TT Hardware Access" before using.

```bash
# Verify device works
python -c "import ttnn; d = ttnn.open_device(device_id=2); print('OK'); ttnn.close_device(d)"
```
