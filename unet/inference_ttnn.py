"""
UNet inference on Tenstorrent hardware using TT-NN.

Usage:
    cd ~/tenstorrent
    source .venv/bin/activate
    python -m unet.inference_ttnn --device_id 0
"""

import argparse
import torch
import torch.nn as nn
import ttnn
from pathlib import Path
import numpy as np

from unet.model import UNetVGG19
from unet.dataset import SegmentationDataModule, get_val_transforms
from common.metrics import compute_pcc


def load_pytorch_model(checkpoint_path: str) -> UNetVGG19:
    """Load trained PyTorch model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'pytorch-lightning_version' in checkpoint:
        # Lightning checkpoint
        model = UNetVGG19.load_from_checkpoint(checkpoint_path)
    else:
        # Simple torch.save checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        model = UNetVGG19(
            num_classes=hparams.get('num_classes', 1),
            pretrained=hparams.get('pretrained', False),
            freeze_encoder=hparams.get('freeze_encoder', False),
            learning_rate=hparams.get('learning_rate', 1e-4),
            bilinear=hparams.get('bilinear', True),
        )
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    return model


def run_inference_pytorch(model: UNetVGG19, images: torch.Tensor) -> torch.Tensor:
    """Run inference using PyTorch (for comparison)."""
    with torch.no_grad():
        return model(images)


def run_conv2d_ttnn(x, conv_module, device):
    """Run a single Conv2d layer on TT-NN (fallback to CPU for complex cases)."""
    # For now, run Conv2d on CPU and transfer to TT for linear ops
    # This is because ttnn.conv2d requires specific setup
    weight = conv_module.weight.data
    bias = conv_module.bias.data if conv_module.bias is not None else None

    out = torch.nn.functional.conv2d(
        x, weight, bias,
        stride=conv_module.stride,
        padding=conv_module.padding,
    )
    return out


def run_inference_ttnn(model: UNetVGG19, images: torch.Tensor, device) -> torch.Tensor:
    """
    Run inference using TT-NN on Tenstorrent hardware.

    Note: UNet with Conv2d layers is complex. This implementation uses a hybrid
    approach: Conv layers on CPU, BatchNorm/ReLU/Linear on TT-NN where possible.

    For full TT-NN implementation, consider using ttnn.conv2d with proper setup.
    """
    # For UNet, we'll use PyTorch for conv layers and TT-NN for final processing
    # This is a hybrid approach - full TT-NN would require more complex setup

    with torch.no_grad():
        # Run encoder on CPU (VGG19 conv layers)
        e1 = model.enc1(images)
        e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2))
        e4 = model.enc4(model.pool3(e3))
        e5 = model.enc5(model.pool4(e4))

        # Bridge
        b = model.bridge(model.pool4(e5))

        # Decoder (can be partially on TT-NN for matmul operations)
        d5 = model.up5(b, e5)
        d4 = model.up4(d5, e4)
        d3 = model.up3(d4, e3)
        d2 = model.up2(d3, e2)
        d1 = model.up1(d2, e1)

        # Final 1x1 convolution - this is essentially a per-pixel linear layer
        # We can run this on TT-NN
        batch_size, channels, height, width = d1.shape

        # Reshape to [batch*height*width, channels] for matmul
        x_flat = d1.permute(0, 2, 3, 1).reshape(-1, channels)

        # Get final conv weights (1x1 conv is equivalent to linear)
        weight = model.final.weight.data.squeeze()  # [out_ch, in_ch]
        bias = model.final.bias.data if model.final.bias is not None else torch.zeros(model.num_classes)

        # Convert to TT-NN tensors
        x_ttnn = ttnn.from_torch(
            x_flat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        weight_ttnn = ttnn.from_torch(
            weight.T,  # Transpose for matmul
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        bias_ttnn = ttnn.from_torch(
            bias.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        # Matmul on TT-NN
        out_ttnn = ttnn.matmul(x_ttnn, weight_ttnn)
        out_ttnn = ttnn.add(out_ttnn, bias_ttnn)

        # Convert back to PyTorch
        out_flat = ttnn.to_torch(out_ttnn)

        # Reshape back to image format
        output = out_flat.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)

    return output


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()

    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    return dice.item()


def main():
    parser = argparse.ArgumentParser(description="UNet inference on Tenstorrent")
    parser.add_argument("--device_id", type=int, default=0, help="TT device ID (0-7)")
    parser.add_argument("--checkpoint", type=str, default="weights_unet/best_model.ckpt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=8, help="Number of test samples")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"UNet Inference on Tenstorrent Device {args.device_id}")
    print(f"{'='*60}\n")

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first: python -m unet.train")
        return

    # Load PyTorch model
    print(f"Loading model from {args.checkpoint}...")
    model = load_pytorch_model(args.checkpoint)
    print(f"Model loaded. Parameters: {model.get_num_parameters():,}")

    # Load test data
    print("Loading test data...")
    dm = SegmentationDataModule(data_dir="./data", batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # Get a batch of test images
    images, masks = next(iter(test_loader))
    images = images[:args.num_samples]
    masks = masks[:args.num_samples]
    print(f"Test batch: {images.shape}")

    # Run PyTorch inference
    print("\nRunning PyTorch inference...")
    pytorch_output = run_inference_pytorch(model, images)
    pytorch_dice = compute_dice(pytorch_output, masks)
    print(f"PyTorch Dice: {pytorch_dice:.4f}")

    # Open TT device
    print(f"\nOpening Tenstorrent device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)
    print("Device opened!")

    try:
        # Run TT-NN inference
        print("Running TT-NN inference (hybrid: Conv on CPU, final layer on TT)...")
        ttnn_output = run_inference_ttnn(model, images, device)
        ttnn_dice = compute_dice(ttnn_output, masks)
        print(f"TT-NN Dice: {ttnn_dice:.4f}")

        # Compare outputs
        pcc = compute_pcc(pytorch_output, ttnn_output)

        print(f"\n{'='*60}")
        print("Results Comparison")
        print(f"{'='*60}")
        print(f"PyTorch Dice:  {pytorch_dice:.4f}")
        print(f"TT-NN Dice:    {ttnn_dice:.4f}")
        print(f"PCC:           {pcc:.6f}")
        print(f"PCC > 0.99:    {'YES' if pcc > 0.99 else 'NO'}")
        print(f"{'='*60}\n")

    finally:
        print("Closing device...")
        ttnn.close_device(device)
        print("Done!")


if __name__ == "__main__":
    main()
