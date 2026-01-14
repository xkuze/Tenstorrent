"""
MNIST inference on Tenstorrent hardware using TT-NN.

================================================================================
DEVICE SELECTION
================================================================================

Available devices: 0, 1, 2, 3 (local), 4, 5, 6, 7 (remote)

1. First write in Teams chat "TT Hardware Access":
   "I'll use device 0" (or another number)

2. Run the script with the desired device_id:
   python -m mnist.inference_ttnn --device_id 0

3. Check available devices:
   python -c "import ttnn; print(ttnn.get_device_ids())"

================================================================================
USAGE
================================================================================

    cd ~/tenstorrent
    source .venv/bin/activate
    python -m mnist.inference_ttnn --device_id 0
"""

import argparse
import torch
import ttnn
from pathlib import Path

from mnist.model import MLP
from mnist.utils import MNISTDataModule
from common.metrics import compute_pcc


def load_pytorch_model(checkpoint_path: str) -> MLP:
    """Load trained PyTorch model from checkpoint."""
    model = MLP.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def run_inference_pytorch(model: MLP, images: torch.Tensor) -> torch.Tensor:
    """Run inference using PyTorch (for comparison)."""
    with torch.no_grad():
        return model(images)


def run_inference_ttnn(model: MLP, images: torch.Tensor, device) -> torch.Tensor:
    """Run inference using TT-NN on Tenstorrent hardware."""

    # Flatten images: [batch, 1, 28, 28] -> [batch, 784]
    x = images.view(images.size(0), -1)

    # Convert input to ttnn tensor
    x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get weights from PyTorch model and run through layers
    layer_idx = 0
    for module in model.network:
        if isinstance(module, torch.nn.Linear):
            # Get weights and bias
            weight = module.weight.data
            bias = module.bias.data

            # Convert to ttnn
            weight_ttnn = ttnn.from_torch(
                weight.T,  # Transpose for ttnn matmul
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

            # Matrix multiplication + bias
            x_ttnn = ttnn.matmul(x_ttnn, weight_ttnn)
            x_ttnn = ttnn.add(x_ttnn, bias_ttnn)

            layer_idx += 1

        elif isinstance(module, torch.nn.ReLU):
            x_ttnn = ttnn.relu(x_ttnn)

        elif isinstance(module, torch.nn.Dropout):
            # Skip dropout during inference
            pass

    # Convert back to PyTorch tensor
    output = ttnn.to_torch(x_ttnn)

    return output


def main():
    parser = argparse.ArgumentParser(description="MNIST inference on Tenstorrent")
    parser.add_argument("--device_id", type=int, default=0, help="TT device ID (0-3)")
    parser.add_argument("--checkpoint", type=str, default="weights_mnist/best_model.ckpt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MNIST Inference on Tenstorrent Device {args.device_id}")
    print(f"{'='*60}\n")

    # Load PyTorch model
    print(f"Loading model from {args.checkpoint}...")
    model = load_pytorch_model(args.checkpoint)
    print(f"Model loaded. Parameters: {model.get_num_parameters():,}")

    # Load test data
    print("Loading test data...")
    dm = MNISTDataModule(data_dir="./data", batch_size=args.batch_size)
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
        print("Running TT-NN inference...")
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
        # Always close device
        print("Closing device...")
        ttnn.close_device(device)
        print("Done!")


if __name__ == "__main__":
    main()
