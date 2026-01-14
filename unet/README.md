# UNet

Image segmentation for pet detection using UNet with VGG19 encoder.

## Structure

```
unet/
├── model.py          # UNetVGG19 LightningModule
├── train.py          # Training script
├── dataset.py        # Oxford-IIIT Pet DataModule
├── inference_ttnn.py # Inference on TensTorrent
├── benchmark.py      # Performance comparison PyTorch vs TT-NN
├── weights/          # Model checkpoints
└── logs/             # TensorBoard and CSV logs
```

## Architecture

- **Encoder**: VGG19 pretrained (frozen weights)
- **Decoder**: Transposed convolutions with skip connections
- **Output**: Binary segmentation mask (pet vs background)

```
Input (3, 128, 128)
    │
    ▼
┌─────────────────┐
│  VGG19 Encoder  │  ← Frozen pretrained weights
│  (5 stages)     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Bridge Layer   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Decoder        │  ← Trainable
│  (5 UpBlocks)   │
└─────────────────┘
    │
    ▼
Output (1, 128, 128)
```

## Training

```bash
python -m unet.train
```

- Dataset: Oxford-IIIT Pet (3680 train / 740 val / 369 test)
- Loss: Binary Cross Entropy
- Optimizer: Adam (lr=1e-4)
- Epochs: 50 with early stopping (patience=10)

## Inference

```bash
# PyTorch inference
python -m unet.inference_ttnn --device_id 3

# With custom batch size
python -m unet.inference_ttnn --device_id 3 --batch_size 8
```

Hybrid approach: Conv layers on CPU, final linear layer on TensTorrent.

## Benchmark

```bash
python -m unet.benchmark --device_id 3 --batch_size 4 --num_runs 10
```

Compares PyTorch CPU vs TT-NN hybrid execution time.

## Results

| Metric | Value |
|--------|-------|
| Test Dice | 0.9310 |
| Test IoU | 0.8723 |
| Test Loss | 0.1266 |

## Files

| File | Description |
|------|-------------|
| `model.py` | UNetVGG19 class with encoder/decoder/skip connections |
| `dataset.py` | SegmentationDataset, OxfordIIITPetDataModule |
| `train.py` | Training loop with callbacks and logging |
| `inference_ttnn.py` | TT-NN inference with PCC comparison |
| `benchmark.py` | Timing comparison between PyTorch and TT-NN |
