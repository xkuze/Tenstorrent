"""
TT-NN Visualizer Profiling Script.

Generates memory and performance reports for UNet model.

Usage:
    # Memory report (requires TTNN_CONFIG_PATH)
    export TTNN_CONFIG_PATH=configs/vis_config.json
    python scripts/run_visualizer_profiling.py --mode memory --device_id 3

    # Performance report (requires TT_METAL_DEVICE_PROFILER)
    TT_METAL_DEVICE_PROFILER=1 python scripts/run_visualizer_profiling.py --mode performance --device_id 3

Output:
    - generated/visualizer/memory_reports/
    - generated/visualizer/profiler_logs/
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_memory_profiling(device_id: int):
    """Run inference with memory profiling enabled."""
    import torch
    import ttnn
    from unet.model import UNetVGG19
    from unet.dataset import OxfordIIITPetDataModule

    print("=== Memory Profiling ===")
    print(f"Config path: {os.environ.get('TTNN_CONFIG_PATH', 'Not set')}")

    # Load model
    checkpoint = Path(__file__).parent.parent / "unet/weights/best_model.ckpt"
    model = UNetVGG19.load_from_checkpoint(str(checkpoint))
    model.eval()
    print(f"Model loaded from {checkpoint}")

    # Load test data
    data_module = OxfordIIITPetDataModule(batch_size=4)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    images, masks = next(iter(test_loader))
    print(f"Test batch: {images.shape}")

    # Open device
    print(f"\nOpening device {device_id}...")
    device = ttnn.open_device(device_id)

    try:
        # Run inference with profiling
        with torch.no_grad():
            # PyTorch forward
            pytorch_out = model(images)

            # Flatten for simple TTNN ops
            batch_size = images.shape[0]
            flat_input = images.view(batch_size, -1)
            weights = model.final.weight.view(model.final.out_channels, -1)
            bias = model.final.bias

            # TTNN operations (these get profiled)
            tt_input = ttnn.from_torch(flat_input, dtype=ttnn.bfloat16, device=device)
            tt_weights = ttnn.from_torch(weights.T, dtype=ttnn.bfloat16, device=device)
            tt_bias = ttnn.from_torch(bias.unsqueeze(0), dtype=ttnn.bfloat16, device=device)

            tt_out = ttnn.matmul(tt_input, tt_weights)
            tt_out = ttnn.add(tt_out, tt_bias)

            result = ttnn.to_torch(tt_out)

        print("Memory profiling complete!")
        print("Check generated/visualizer/memory_reports/ for output")

    finally:
        ttnn.close_device(device)
        print("Device closed.")


def run_performance_profiling(device_id: int):
    """Run inference with performance profiling enabled."""
    import torch
    import ttnn
    from unet.model import UNetVGG19
    from unet.dataset import OxfordIIITPetDataModule

    print("=== Performance Profiling ===")
    print(f"TT_METAL_DEVICE_PROFILER: {os.environ.get('TT_METAL_DEVICE_PROFILER', 'Not set')}")

    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        print("Warning: TT_METAL_DEVICE_PROFILER not set to 1")
        print("Run with: TT_METAL_DEVICE_PROFILER=1 python scripts/run_visualizer_profiling.py --mode performance")

    # Load model
    checkpoint = Path(__file__).parent.parent / "unet/weights/best_model.ckpt"
    model = UNetVGG19.load_from_checkpoint(str(checkpoint))
    model.eval()
    print(f"Model loaded from {checkpoint}")

    # Load test data
    data_module = OxfordIIITPetDataModule(batch_size=4)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    images, masks = next(iter(test_loader))
    print(f"Test batch: {images.shape}")

    # Open device
    print(f"\nOpening device {device_id}...")
    device = ttnn.open_device(device_id)

    try:
        with torch.no_grad():
            # PyTorch forward
            pytorch_out = model(images)

            # Simple TTNN ops for profiling
            batch_size = images.shape[0]
            flat_input = images.view(batch_size, -1)
            weights = model.final.weight.view(model.final.out_channels, -1)
            bias = model.final.bias

            tt_input = ttnn.from_torch(flat_input, dtype=ttnn.bfloat16, device=device)
            tt_weights = ttnn.from_torch(weights.T, dtype=ttnn.bfloat16, device=device)
            tt_bias = ttnn.from_torch(bias.unsqueeze(0), dtype=ttnn.bfloat16, device=device)

            tt_out = ttnn.matmul(tt_input, tt_weights)
            tt_out = ttnn.add(tt_out, tt_bias)

            result = ttnn.to_torch(tt_out)

        print("Performance profiling complete!")
        print("Check generated/visualizer/profiler_logs/ for output")

    finally:
        ttnn.close_device(device)
        print("Device closed.")


def main():
    parser = argparse.ArgumentParser(description="TT-NN Visualizer Profiling")
    parser.add_argument("--mode", choices=["memory", "performance"], required=True)
    parser.add_argument("--device_id", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "memory":
        run_memory_profiling(args.device_id)
    else:
        run_performance_profiling(args.device_id)


if __name__ == "__main__":
    main()
