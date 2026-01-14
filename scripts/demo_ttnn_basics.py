"""
TT-NN Basics Demo Script.

Demonstrates key TT-NN concepts for learning and evaluation:
- Tensor layouts (TILE, ROW_MAJOR)
- Memory management (L1, DRAM)
- PyTorch to TT-NN conversion patterns
- Common API patterns

Usage:
    python scripts/demo_ttnn_basics.py --device_id 3
    python scripts/demo_ttnn_basics.py --device_id 3 --section layouts
    python scripts/demo_ttnn_basics.py --device_id 3 --section conversion
    python scripts/demo_ttnn_basics.py --device_id 3 --section api

Covers Excel tasks:
- Row 5: Tensor Layout and Memory Management
- Row 8: Framework Translation Challenges
- Row 10: Model Conversion Workflow
- Row 13: API Learning Curve
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import ttnn

from common.metrics import compute_pcc


def demo_tensor_layouts(device):
    """
    Demo: Tensor Layout and Memory Management (Row 5).

    TT-NN supports different tensor layouts:
    - TILE: 32x32 tiles, optimal for matrix operations
    - ROW_MAJOR: Standard row-major layout

    Memory types:
    - DRAM: Large capacity, higher latency
    - L1: Fast local memory, limited capacity
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Tensor Layouts and Memory Management")
    print("=" * 70)

    # Create PyTorch tensor
    x = torch.randn(1, 1, 64, 64)
    print(f"\nPyTorch tensor: shape={x.shape}, dtype={x.dtype}")

    # 1. Default layout (TILE) in DRAM
    print("\n--- Layout: TILE (default), Memory: DRAM ---")
    tt_tile = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(f"  TTNN tensor layout: {tt_tile.layout}")
    print("  Memory config: DRAM")
    print(f"  Shape: {tt_tile.shape}")

    # 2. ROW_MAJOR layout
    print("\n--- Layout: ROW_MAJOR ---")
    tt_row = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    print(f"  TTNN tensor layout: {tt_row.layout}")
    print(f"  Shape: {tt_row.shape}")

    # 3. Convert between layouts
    print("\n--- Layout Conversion ---")
    tt_converted = ttnn.to_layout(tt_row, ttnn.TILE_LAYOUT)
    print(f"  ROW_MAJOR -> TILE: {tt_converted.layout}")

    # 4. L1 Memory (faster but limited)
    print("\n--- Memory: L1 (faster, limited capacity) ---")
    tt_l1 = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print("  Memory config: L1")
    print(f"  Shape: {tt_l1.shape}")

    # Verify correctness
    x_back = ttnn.to_torch(tt_tile)
    pcc = compute_pcc(x, x_back.float())
    print(f"\n✅ Round-trip PCC: {pcc:.6f}")

    print("\n📝 Key Takeaways:")
    print("  - TILE_LAYOUT (32x32): Best for matmul, conv2d")
    print("  - ROW_MAJOR_LAYOUT: For element-wise ops, reshaping")
    print("  - DRAM: Large tensors, weights")
    print("  - L1: Intermediate activations, frequently accessed data")


def demo_pytorch_conversion(device):
    """
    Demo: Framework Translation / PyTorch to TT-NN Conversion (Rows 8, 10).

    Shows common patterns for converting PyTorch code to TT-NN.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: PyTorch to TT-NN Conversion Patterns")
    print("=" * 70)

    # Pattern 1: Simple tensor operations
    print("\n--- Pattern 1: Tensor Transfer ---")
    print("  PyTorch: x = torch.randn(1, 64)")
    print("  TT-NN:   tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)")

    x = torch.randn(1, 64)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    x_back = ttnn.to_torch(tt_x)
    print(f"  ✅ Transfer PCC: {compute_pcc(x, x_back.float()):.6f}")

    # Pattern 2: Matrix multiplication (requires TILE layout, shapes multiple of 32)
    print("\n--- Pattern 2: Matrix Multiplication ---")
    print("  PyTorch: y = torch.matmul(a, b)")
    print("  TT-NN:   tt_y = ttnn.matmul(tt_a, tt_b)  # TILE layout required")

    a = torch.randn(1, 1, 32, 64)
    b = torch.randn(1, 1, 64, 32)
    pytorch_out = torch.matmul(a, b)

    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.matmul(tt_a, tt_b)
    ttnn_out = ttnn.to_torch(tt_out)

    pcc = compute_pcc(pytorch_out, ttnn_out.float())
    print(f"  ✅ Matmul PCC: {pcc:.6f}")

    # Pattern 3: Element-wise operations
    print("\n--- Pattern 3: Element-wise Operations ---")
    print("  PyTorch: y = torch.relu(x)")
    print("  TT-NN:   tt_y = ttnn.relu(tt_x)")

    x = torch.randn(1, 1, 32, 32)
    pytorch_relu = torch.relu(x)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_relu = ttnn.relu(tt_x)
    ttnn_relu = ttnn.to_torch(tt_relu)

    pcc = compute_pcc(pytorch_relu, ttnn_relu.float())
    print(f"  ✅ ReLU PCC: {pcc:.6f}")

    # Pattern 4: Add operation
    print("\n--- Pattern 4: Binary Operations ---")
    print("  PyTorch: c = a + b")
    print("  TT-NN:   tt_c = ttnn.add(tt_a, tt_b)")

    a = torch.randn(1, 64)
    b = torch.randn(1, 64)
    pytorch_add = a + b

    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, device=device)
    tt_add = ttnn.add(tt_a, tt_b)
    ttnn_add = ttnn.to_torch(tt_add)

    pcc = compute_pcc(pytorch_add, ttnn_add.float())
    print(f"  ✅ Add PCC: {pcc:.6f}")

    # Pattern 5: Linear layer conversion (simplified for TILE_LAYOUT compatibility)
    print("\n--- Pattern 5: Linear Layer Conversion ---")
    print("  PyTorch: linear = nn.Linear(64, 32); y = linear(x)")
    print("  TT-NN:   tt_y = ttnn.matmul(tt_x, tt_weights) + tt_bias")
    print("  Note: TILE_LAYOUT requires 4D tensors with dims multiple of 32")

    # Use 4D shapes compatible with TILE_LAYOUT (dims must be multiples of 32)
    linear = torch.nn.Linear(64, 32)
    x = torch.randn(1, 1, 32, 64)  # Batch of 32 vectors, each 64-dim

    with torch.no_grad():
        # Flatten for PyTorch linear, then reshape back
        x_flat = x.view(-1, 64)  # (32, 64)
        pytorch_out = linear(x_flat)  # (32, 32)
        pytorch_linear = pytorch_out.view(1, 1, 32, 32)

    # TTNN conversion with TILE_LAYOUT
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    # Weight: (64, 32) -> (1, 1, 64, 32) for matmul
    weight_4d = linear.weight.T.unsqueeze(0).unsqueeze(0)
    tt_w = ttnn.from_torch(
        weight_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_out = ttnn.matmul(tt_x, tt_w)
    # Add bias via broadcast (bias shape: 1, 1, 1, 32)
    bias_4d = linear.bias.view(1, 1, 1, 32).expand(1, 1, 32, 32)
    tt_b = ttnn.from_torch(
        bias_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_out = ttnn.add(tt_out, tt_b)
    ttnn_linear = ttnn.to_torch(tt_out)

    pcc = compute_pcc(pytorch_linear, ttnn_linear.float())
    print(f"  ✅ Linear PCC: {pcc:.6f}")

    print("\n📝 Key Conversion Patterns:")
    print("  - torch.matmul → ttnn.matmul")
    print("  - torch.relu → ttnn.relu")
    print("  - torch.add / + → ttnn.add")
    print("  - nn.Linear → ttnn.matmul + ttnn.add")
    print("  - Weight transpose: linear.weight.T for matmul")


def demo_api_patterns(device):
    """
    Demo: API Learning Curve / Common Patterns (Row 13).

    Shows best practices and common API patterns.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: TT-NN API Patterns and Best Practices")
    print("=" * 70)

    # Pattern 1: Device context management
    print("\n--- Pattern 1: Device Context Management ---")
    print("  # Open device at start")
    print("  device = ttnn.open_device(device_id)")
    print("  try:")
    print("      # ... operations ...")
    print("  finally:")
    print("      ttnn.close_device(device)")
    print("  ✅ Always use try/finally for device cleanup")

    # Pattern 2: Data type selection
    print("\n--- Pattern 2: Data Type Selection ---")
    print("  bfloat16: Default for most operations (good accuracy, fast)")
    print("  float32:  When high precision needed")
    print("  uint16:   For indices, embeddings")

    x = torch.randn(1, 32)
    tt_bf16 = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    print(f"  ✅ Created bfloat16 tensor: {tt_bf16.dtype}")

    # Pattern 3: Reshape operations
    print("\n--- Pattern 3: Reshape Operations ---")
    print("  PyTorch: x.view(batch, -1)")
    print("  TT-NN:   ttnn.reshape(tt_x, [batch, new_size])")

    x = torch.randn(2, 4, 8, 8)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    tt_reshaped = ttnn.reshape(tt_x, [2, 256])
    print(f"  ✅ Reshaped: {x.shape} -> {tt_reshaped.shape}")

    # Pattern 4: Batch processing
    print("\n--- Pattern 4: Batch Processing ---")
    print("  TT-NN works best with batched operations")
    print("  Pad batch to power of 2 for efficiency")

    batch_sizes = [1, 4, 8, 16, 32]
    print(f"  Recommended batch sizes: {batch_sizes}")

    # Pattern 5: Operation fusion hint
    print("\n--- Pattern 5: Operation Chaining ---")
    print("  # Chain operations without returning to host")
    print("  tt_x = ttnn.from_torch(x, device=device)")
    print("  tt_x = ttnn.relu(tt_x)        # stays on device")
    print("  tt_x = ttnn.matmul(tt_x, w)   # stays on device")
    print("  result = ttnn.to_torch(tt_x)  # only transfer at end")

    x = torch.randn(1, 1, 32, 64)
    w = torch.randn(1, 1, 64, 32)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Chain operations
    tt_x = ttnn.relu(tt_x)
    tt_out = ttnn.matmul(tt_x, tt_w)
    result = ttnn.to_torch(tt_out)

    print(f"  ✅ Chained ops result shape: {result.shape}")

    # Pattern 6: Config for profiling
    print("\n--- Pattern 6: Profiling Configuration ---")
    print("  # For memory profiling:")
    print("  export TTNN_CONFIG_PATH=configs/vis_config.json")
    print("  # For performance profiling:")
    print("  TT_METAL_DEVICE_PROFILER=1 python script.py")

    print("\n📝 Best Practices Summary:")
    print("  1. Use bfloat16 for most operations")
    print("  2. Keep data on device, minimize transfers")
    print("  3. Use TILE_LAYOUT for compute-heavy ops")
    print("  4. Batch operations when possible")
    print("  5. Always close device in finally block")


def main():
    parser = argparse.ArgumentParser(description="TT-NN Basics Demo")
    parser.add_argument("--device_id", type=int, default=3, help="Device ID")
    parser.add_argument(
        "--section",
        choices=["all", "layouts", "conversion", "api"],
        default="all",
        help="Which section to run",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TT-NN BASICS DEMONSTRATION")
    print("=" * 70)
    print(f"Device ID: {args.device_id}")

    device = ttnn.open_device(device_id=args.device_id)

    try:
        if args.section in ["all", "layouts"]:
            demo_tensor_layouts(device)

        if args.section in ["all", "conversion"]:
            demo_pytorch_conversion(device)

        if args.section in ["all", "api"]:
            demo_api_patterns(device)

        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETE")
        print("=" * 70)

    finally:
        ttnn.close_device(device)
        print("\nDevice closed.")


if __name__ == "__main__":
    main()
