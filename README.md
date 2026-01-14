# TensTorrent ML Project

ML project for training and inference on TensTorrent hardware using PyTorch Lightning and TT-NN.

## Table of Contents

- [Tasks](#tasks)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference on TensTorrent](#inference-on-tenstorrent)
- [Evaluation Scripts](#evaluation-scripts)
- [Profiling](#profiling)
- [Device Selection](#device-selection)
- [Tools](#tools)
- [Documentation](#documentation)
- [Module READMEs](#module-readmes)

## Tasks

| Task | Architecture | Dataset | Test Metric |
|------|--------------|---------|-------------|
| MNIST | MLP | 70k images | **97.76%** accuracy |
| CIFAR-10 | CNN | 60k images | **75.36%** accuracy |
| UNet | VGG19 encoder | Oxford-IIIT Pet | **0.93** Dice score |

## Quick Start

```bash
# SSH to server (requires VPN)
ssh ekaterina_kuzmina1@10.30.0.207

# Activate environment
cd ~/tenstorrent
source .venv/bin/activate

# Verify TT-NN installation
python scripts/sanity_check.py --device_id 3

# Run inference
python -m mnist.inference_ttnn --device_id 3
python -m cifar.inference_ttnn --device_id 3
python -m unet.inference_ttnn --device_id 3
```

## Project Structure

```
tenstorrent/
├── mnist/                # Digit classification (MLP)
│   ├── model.py              # MLP LightningModule
│   ├── train.py              # Optuna hyperparameter search
│   ├── inference_ttnn.py     # TT-NN inference
│   ├── weights/              # Checkpoints
│   └── logs/                 # Training logs
│
├── cifar/                # Image classification (CNN)
│   ├── model.py              # CNN LightningModule
│   ├── train.py              # Optuna hyperparameter search
│   ├── inference_ttnn.py     # Hybrid inference (Conv CPU + FC TT-NN)
│   ├── weights/
│   └── logs/
│
├── unet/                 # Segmentation (VGG19 encoder)
│   ├── model.py              # UNetVGG19 LightningModule
│   ├── dataset.py            # Oxford-IIIT Pet DataModule
│   ├── train.py              # Training with early stopping
│   ├── inference_ttnn.py     # TT-NN inference
│   ├── benchmark.py          # PyTorch vs TT-NN timing
│   ├── weights/
│   └── logs/
│
├── common/               # Shared utilities
│   └── metrics.py            # compute_pcc for output validation
│
├── scripts/              # Evaluation and demo scripts
│   ├── sanity_check.py       # Environment verification
│   ├── demo_ttnn_basics.py   # TT-NN API tutorial
│   ├── check_ttnn_ops.py     # Operation support checker
│   ├── run_tracer.py         # Model tracing and codegen
│   └── run_visualizer_profiling.py  # Memory/performance profiling
│
├── configs/              # Configuration files
│   └── vis_config.json       # TT-NN Visualizer config
│
├── generated/            # Auto-generated outputs
│   ├── tracer/               # Traced graphs, tests, codegen
│   └── visualizer/           # Memory and performance reports
│
└── data/                 # Datasets (gitignored)
```

## Training

All models use PyTorch Lightning with automatic checkpointing and early stopping.

```bash
# MNIST - MLP with Optuna (20 trials)
python -m mnist.train

# CIFAR-10 - CNN with Optuna (20 trials)
python -m cifar.train

# UNet - VGG19 encoder (50 epochs, early stopping)
python -m unet.train
```

## Inference on TensTorrent

```bash
# Basic inference with PCC validation
python -m mnist.inference_ttnn --device_id 3
python -m cifar.inference_ttnn --device_id 3
python -m unet.inference_ttnn --device_id 3

# UNet benchmark (timing comparison)
python -m unet.benchmark --device_id 3 --batch_size 4 --num_runs 10
```

## Evaluation Scripts

```bash
# Verify environment
python scripts/sanity_check.py --device_id 3

# Learn TT-NN basics (layouts, conversion patterns, API)
python scripts/demo_ttnn_basics.py --device_id 3

# Check which PyTorch ops have TT-NN equivalents
python scripts/check_ttnn_ops.py

# Trace model and generate tests/codegen
python scripts/run_tracer.py --phase all

# Run auto-generated unit tests
pytest generated/tracer/tests/test_unet_ops.py -v
```

## Profiling

```bash
# Memory profiling
export TTNN_CONFIG_PATH=configs/vis_config.json
python scripts/run_visualizer_profiling.py --mode memory --device_id 3

# Performance profiling
TT_METAL_DEVICE_PROFILER=1 python scripts/run_visualizer_profiling.py \
    --mode performance --device_id 3

# View reports
uv run ttnn-visualizer  # Opens http://localhost:8000
```

## Tools

| Tool | Purpose |
|------|---------|
| **uv** | Package manager |
| **ruff** | Linting and formatting |
| **PyTorch Lightning** | Training framework |
| **Optuna** | Hyperparameter optimization |
| **TT-NN** | TensTorrent inference library |
| **ttnn-visualizer** | Memory/performance visualization |


## Module READMEs

Each module has its own README with detailed documentation:

- [mnist/README.md](mnist/README.md) - MLP architecture, Optuna search space
- [cifar/README.md](cifar/README.md) - CNN architecture, hybrid inference
- [unet/README.md](unet/README.md) - VGG19 encoder, segmentation results
- [common/README.md](common/README.md) - PCC metric, thresholds
- [scripts/README.md](scripts/README.md) - All evaluation scripts
- [configs/README.md](configs/README.md) - Visualizer configuration
- [generated/README.md](generated/README.md) - Tracer and visualizer outputs
