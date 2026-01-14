"""
TT-NN Environment Sanity Check.

Verifies that TT-NN is properly installed and basic operations work.

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --device_id 3

Output:
    - Environment verification
    - Basic TTNN operation test
    - Device connectivity test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required packages can be imported."""
    print("=" * 60)
    print("1. Checking imports...")
    print("=" * 60)

    packages = [
        ("torch", "PyTorch"),
        ("ttnn", "TT-NN"),
        ("lightning", "PyTorch Lightning"),
        ("torchvision", "TorchVision"),
    ]

    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False

    return all_ok


def check_ttnn_version():
    """Check TTNN version and config."""
    print("\n" + "=" * 60)
    print("2. Checking TT-NN version...")
    print("=" * 60)

    import ttnn

    print(f"  TTNN version: {getattr(ttnn, '__version__', 'unknown')}")
    print(f"  Cache path: {ttnn.CONFIG.cache_path}")
    print(f"  Fast runtime: {ttnn.CONFIG.enable_fast_runtime_mode}")

    return True


def check_devices():
    """Check available TT devices."""
    print("\n" + "=" * 60)
    print("3. Checking available devices...")
    print("=" * 60)

    import ttnn

    try:
        device_ids = ttnn.get_device_ids()
        print(f"  Available devices: {device_ids}")
        return device_ids
    except Exception as e:
        print(f"  ❌ Error getting devices: {e}")
        return []


def check_device_connectivity(device_id: int):
    """Test opening and closing a device."""
    print("\n" + "=" * 60)
    print(f"4. Testing device {device_id} connectivity...")
    print("=" * 60)

    import ttnn

    try:
        print(f"  Opening device {device_id}...")
        device = ttnn.open_device(device_id=device_id)
        print("  ✅ Device opened successfully")

        ttnn.close_device(device)
        print("  ✅ Device closed successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_basic_operations(device_id: int):
    """Test basic TTNN operations."""
    print("\n" + "=" * 60)
    print(f"5. Testing basic TTNN operations on device {device_id}...")
    print("=" * 60)

    import torch
    import ttnn

    try:
        device = ttnn.open_device(device_id=device_id)

        # Test 1: Tensor transfer
        print("  Testing tensor transfer...")
        x = torch.randn(1, 32, 32)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        x_back = ttnn.to_torch(tt_x)
        print(f"    ✅ from_torch/to_torch: shape {x.shape} -> {x_back.shape}")

        # Test 2: Basic math
        print("  Testing basic operations...")
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, device=device)
        tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, device=device)
        tt_c = ttnn.add(tt_a, tt_b)
        c = ttnn.to_torch(tt_c)
        print(f"    ✅ ttnn.add: {a.shape} + {b.shape} = {c.shape}")

        # Test 3: Matmul (requires TILE layout, shapes must be multiples of 32)
        print("  Testing matmul...")
        m1 = torch.randn(1, 1, 32, 64)
        m2 = torch.randn(1, 1, 64, 32)
        tt_m1 = ttnn.from_torch(
            m1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_m2 = ttnn.from_torch(
            m2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_out = ttnn.matmul(tt_m1, tt_m2)
        out = ttnn.to_torch(tt_out)
        print(f"    ✅ ttnn.matmul: {m1.shape} @ {m2.shape} = {out.shape}")

        # Test 4: Activation (requires TILE layout)
        print("  Testing activation...")
        relu_input = torch.randn(1, 1, 32, 32)
        tt_relu_in = ttnn.from_torch(
            relu_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_relu = ttnn.relu(tt_relu_in)
        relu_out = ttnn.to_torch(tt_relu)
        print(f"    ✅ ttnn.relu: {relu_input.shape} -> {relu_out.shape}")

        # Compute PCC for validation
        from common.metrics import compute_pcc

        expected = torch.relu(relu_input)
        pcc = compute_pcc(expected, relu_out.float())
        print(f"    ✅ ReLU PCC: {pcc:.6f}")

        ttnn.close_device(device)
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        try:
            ttnn.close_device(device)
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="TT-NN Sanity Check")
    parser.add_argument("--device_id", type=int, default=3, help="Device ID to test")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TT-NN ENVIRONMENT SANITY CHECK")
    print("=" * 60)

    results = []

    # 1. Check imports
    results.append(("Imports", check_imports()))

    # 2. Check TTNN version
    results.append(("TTNN Version", check_ttnn_version()))

    # 3. Check devices
    devices = check_devices()
    results.append(("Device Discovery", len(devices) > 0))

    # 4. Check device connectivity
    if args.device_id in devices:
        results.append(("Device Connectivity", check_device_connectivity(args.device_id)))

        # 5. Check basic operations
        results.append(("Basic Operations", check_basic_operations(args.device_id)))
    else:
        print(f"\n⚠️  Device {args.device_id} not available. Available: {devices}")
        results.append(("Device Connectivity", False))
        results.append(("Basic Operations", False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Environment is ready!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
