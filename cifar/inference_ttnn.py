"""
CIFAR-10 CNN inference on Tenstorrent hardware using TT-NN.

Usage:
    cd ~/tenstorrent
    source .venv/bin/activate
    python -m cifar.inference_ttnn --device_id 2
"""

import argparse
import torch
import ttnn
from pathlib import Path

from cifar.model import CNN
from cifar.utils import CIFAR10DataModule
from common.metrics import compute_pcc

MODULE_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = MODULE_DIR / "weights" / "best_model.ckpt"


def load_pytorch_model(checkpoint_path: str) -> CNN:
    """Load trained PyTorch model from checkpoint."""
    model = CNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def run_inference_pytorch(model: CNN, images: torch.Tensor) -> torch.Tensor:
    """Run inference using PyTorch (for comparison)."""
    with torch.no_grad():
        return model(images)


def run_inference_ttnn(model: CNN, images: torch.Tensor, device) -> torch.Tensor:
    """
    Run inference using TT-NN on Tenstorrent hardware.

    Note: CNN with Conv2d is more complex to convert to TT-NN.
    This implementation uses ttnn.conv2d for convolutional layers.
    """
    batch_size = images.shape[0]

    # Convert input to ttnn tensor
    # CIFAR images: [batch, 3, 32, 32]
    x = images

    # Process through conv layers
    for module in model.conv_layers:
        if isinstance(module, torch.nn.Conv2d):
            # Get conv parameters
            weight = module.weight.data  # [out_ch, in_ch, kH, kW]
            bias = module.bias.data if module.bias is not None else None

            # Run conv2d on CPU/GPU for now (ttnn.conv2d requires special setup)
            x = torch.nn.functional.conv2d(
                x, weight, bias,
                stride=module.stride,
                padding=module.padding
            )

        elif isinstance(module, torch.nn.ReLU):
            x = torch.nn.functional.relu(x)

        elif isinstance(module, torch.nn.MaxPool2d):
            x = torch.nn.functional.max_pool2d(
                x,
                kernel_size=module.kernel_size,
                stride=module.stride
            )

        elif isinstance(module, torch.nn.Dropout):
            pass  # Skip during inference

    # Flatten
    x = x.view(batch_size, -1)

    # Convert to ttnn for FC layers (these work well on TT hardware)
    x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Process FC layers on TT-NN
    for module in model.fc_layers:
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            bias = module.bias.data

            weight_ttnn = ttnn.from_torch(
                weight.T,
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

            x_ttnn = ttnn.matmul(x_ttnn, weight_ttnn)
            x_ttnn = ttnn.add(x_ttnn, bias_ttnn)

        elif isinstance(module, torch.nn.ReLU):
            x_ttnn = ttnn.relu(x_ttnn)

        elif isinstance(module, (torch.nn.Dropout, torch.nn.Flatten)):
            pass

    # Convert back to PyTorch
    output = ttnn.to_torch(x_ttnn)

    return output


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 inference on Tenstorrent")
    parser.add_argument("--device_id", type=int, default=0, help="TT device ID (0-3)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=32, help="Number of test samples")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"CIFAR-10 CNN Inference on Tenstorrent Device {args.device_id}")
    print(f"{'='*60}\n")

    # Load PyTorch model
    print(f"Loading model from {args.checkpoint}...")
    model = load_pytorch_model(args.checkpoint)
    print(f"Model loaded. Parameters: {model.get_num_parameters():,}")

    # Load test data
    print("Loading test data...")
    dm = CIFAR10DataModule(data_dir="./data", batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images[:args.num_samples]
    labels = labels[:args.num_samples]
    print(f"Test batch: {images.shape}")

    # Run PyTorch inference
    print("\nRunning PyTorch inference...")
    pytorch_output = run_inference_pytorch(model, images)
    pytorch_preds = torch.argmax(pytorch_output, dim=1)
    pytorch_acc = (pytorch_preds == labels).float().mean().item()
    print(f"PyTorch accuracy: {pytorch_acc * 100:.2f}%")

    # Open TT device
    print(f"\nOpening Tenstorrent device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)
    print("Device opened!")

    try:
        # Run TT-NN inference
        print("Running TT-NN inference (Conv on CPU, FC on TT)...")
        ttnn_output = run_inference_ttnn(model, images, device)
        ttnn_preds = torch.argmax(ttnn_output, dim=1)
        ttnn_acc = (ttnn_preds == labels).float().mean().item()
        print(f"TT-NN accuracy: {ttnn_acc * 100:.2f}%")

        # Compare outputs
        pcc = compute_pcc(pytorch_output, ttnn_output)
        print(f"\n{'='*60}")
        print(f"Results Comparison")
        print(f"{'='*60}")
        print(f"PyTorch accuracy:  {pytorch_acc * 100:.2f}%")
        print(f"TT-NN accuracy:    {ttnn_acc * 100:.2f}%")
        print(f"PCC (correlation): {pcc:.6f}")
        print(f"PCC > 0.99:        {'YES' if pcc > 0.99 else 'NO'}")
        print(f"{'='*60}\n")

    finally:
        print("Closing device...")
        ttnn.close_device(device)
        print("Done!")


if __name__ == "__main__":
    main()
