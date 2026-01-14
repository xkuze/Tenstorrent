"""
Performance benchmark for UNet on Tenstorrent.

Usage:
    cd ~/tenstorrent
    source .venv/bin/activate
    python -m unet.benchmark --device_id 2
"""

import argparse
import time
import torch
import ttnn
from pathlib import Path

from unet.inference_ttnn import (
    load_pytorch_model,
    run_inference_ttnn,
)

MODULE_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = MODULE_DIR / "weights" / "best_model.ckpt"


def benchmark_pytorch(model, images, num_runs=10, warmup=2):
    """Benchmark PyTorch inference."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(images)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(images)
        times.append(time.perf_counter() - start)

    return times


def benchmark_ttnn(model, images, device, num_runs=10, warmup=2):
    """Benchmark TT-NN inference."""
    # Warmup
    for _ in range(warmup):
        _ = run_inference_ttnn(model, images, device)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = run_inference_ttnn(model, images, device)
        times.append(time.perf_counter() - start)

    return times


def print_stats(name, times):
    """Print timing statistics."""
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\n{name}:")
    print(f"  Average: {avg * 1000:.2f} ms")
    print(f"  Min:     {min_t * 1000:.2f} ms")
    print(f"  Max:     {max_t * 1000:.2f} ms")
    print(f"  Throughput: {1 / avg:.2f} inferences/sec")
    return avg


def main():
    parser = argparse.ArgumentParser(description="UNet Performance Benchmark")
    parser.add_argument("--device_id", type=int, default=2, help="TT device ID")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    args = parser.parse_args()

    print("=" * 60)
    print("UNet Performance Benchmark")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Num runs: {args.num_runs}")
    print(f"Device ID: {args.device_id}")

    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"\nCheckpoint not found: {args.checkpoint}")
        return

    # Load model
    print("\nLoading model...")
    model = load_pytorch_model(args.checkpoint)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create test input
    images = torch.randn(args.batch_size, 3, 256, 256)
    print(f"Input shape: {images.shape}")

    # Benchmark PyTorch
    print("\n" + "-" * 40)
    print("Benchmarking PyTorch (CPU)...")
    pytorch_times = benchmark_pytorch(model, images, args.num_runs, args.warmup)
    pytorch_avg = print_stats("PyTorch (CPU)", pytorch_times)

    # Open TT device
    print("\n" + "-" * 40)
    print(f"Opening Tenstorrent device {args.device_id}...")

    try:
        device = ttnn.open_device(device_id=args.device_id)
        print("Device opened!")

        # Benchmark TT-NN
        print("\nBenchmarking TT-NN (hybrid mode)...")
        ttnn_times = benchmark_ttnn(model, images, device, args.num_runs, args.warmup)
        ttnn_avg = print_stats("TT-NN (Hybrid)", ttnn_times)

        # Comparison
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"PyTorch avg:  {pytorch_avg * 1000:.2f} ms")
        print(f"TT-NN avg:    {ttnn_avg * 1000:.2f} ms")

        if ttnn_avg < pytorch_avg:
            speedup = pytorch_avg / ttnn_avg
            print(f"TT-NN is {speedup:.2f}x FASTER than PyTorch")
        else:
            slowdown = ttnn_avg / pytorch_avg
            print(f"TT-NN is {slowdown:.2f}x slower (hybrid mode, Conv on CPU)")

        print("\nNote: Current implementation uses hybrid mode")
        print("(Conv2d on CPU, final matmul on TT). Full TT-NN")
        print("implementation would show better performance.")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTry resetting the device: tt-smi -r <device_id>")

    finally:
        if "device" in locals():
            print("\nClosing device...")
            ttnn.close_device(device)
            print("Done!")


if __name__ == "__main__":
    main()
