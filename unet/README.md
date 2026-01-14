# UNet

Image segmentation for pet detection. VGG19 encoder with frozen pretrained weights.

## Structure

```
unet/
├── model.py          # UNetVGG19 definition
├── train.py          # Training script
├── dataset.py        # Oxford-IIIT Pet DataModule
├── inference_ttnn.py # Inference on TensTorrent
├── benchmark.py      # Performance comparison
├── weights/          # Checkpoints
└── logs/             # Training logs
```

## Training

```bash
python -m unet.train
```

Uses Oxford-IIIT Pet dataset. VGG19 encoder stays frozen, only decoder trains. Binary segmentation — pet vs background.

## Inference

```bash
python -m unet.inference_ttnn --device_id 2
```

Similar hybrid approach as CIFAR. Conv on CPU, final matmul layer on TensTorrent.

## Benchmark

```bash
python -m unet.benchmark --device_id 2 --batch_size 4
```

Compares PyTorch CPU vs TT-NN hybrid timing across multiple runs.

## Results

Dice score: ~0.29 (binary segmentation on test set)
