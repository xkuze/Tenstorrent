"""
Check TTNN operation support for traced PyTorch operations.

Usage:
    python scripts/check_ttnn_ops.py
    python scripts/check_ttnn_ops.py --ops conv2d relu batch_norm

Output:
    - Console report of supported/unsupported operations
    - Updates generated/tracer/metadata/unet_ttnn_ops_mapping.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ttnn


# PyTorch to TTNN operation mapping with workarounds
PYTORCH_TO_TTNN = {
    # Direct mappings
    "conv2d": {"ttnn": "conv2d", "type": "direct"},
    "relu": {"ttnn": "relu", "type": "direct"},
    "batch_norm": {"ttnn": "batch_norm", "type": "direct"},
    "max_pool2d": {"ttnn": "max_pool2d", "type": "direct"},
    "pad": {"ttnn": "pad", "type": "direct"},
    "matmul": {"ttnn": "matmul", "type": "direct"},
    "add": {"ttnn": "add", "type": "direct"},
    "sigmoid": {"ttnn": "sigmoid", "type": "direct"},
    "softmax": {"ttnn": "softmax", "type": "direct"},
    "transpose": {"ttnn": "transpose", "type": "direct"},
    "permute": {"ttnn": "permute", "type": "direct"},
    "squeeze": {"ttnn": "squeeze", "type": "direct"},
    "unsqueeze": {"ttnn": "unsqueeze", "type": "direct"},
    "reshape": {"ttnn": "reshape", "type": "direct"},
    "concat": {"ttnn": "concat", "type": "direct"},
    "upsample": {"ttnn": "upsample", "type": "direct"},
    # Workaround mappings
    "cat": {"ttnn": "concat", "type": "workaround", "note": "Use ttnn.concat instead"},
    "interpolate": {"ttnn": "upsample", "type": "workaround", "note": "Use ttnn.upsample"},
    "flatten": {"ttnn": "reshape", "type": "workaround", "note": "Use ttnn.reshape"},
    # Host-side operations
    "randn": {"ttnn": "from_torch", "type": "host", "note": "Generate on host, transfer"},
}

# UNet operations from tracer
UNET_OPS = {
    "conv2d": 29,
    "relu": 28,
    "batch_norm": 12,
    "max_pool2d": 5,
    "interpolate": 5,
    "pad": 5,
    "cat": 5,
    "randn": 1,
    "flatten": 1,
}


def check_ttnn_support(op_name: str) -> dict:
    """Check if a TTNN operation exists and return status."""
    mapping = PYTORCH_TO_TTNN.get(op_name, {})
    ttnn_op = mapping.get("ttnn", op_name)
    op_type = mapping.get("type", "unknown")
    note = mapping.get("note", "")

    # Check if TTNN has this operation
    exists = hasattr(ttnn, ttnn_op)

    return {
        "pytorch_op": op_name,
        "ttnn_op": ttnn_op,
        "exists": exists,
        "type": op_type,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(description="Check TTNN operation support")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=list(UNET_OPS.keys()),
        help="Operations to check (default: UNet ops)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all known PyTorch operations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated/tracer/metadata/unet_ttnn_ops_mapping.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    ops_to_check = list(PYTORCH_TO_TTNN.keys()) if args.all else args.ops

    print("=" * 60)
    print("TTNN Operation Support Check")
    print("=" * 60)

    results = {"direct": [], "workaround": [], "host": [], "unsupported": []}

    for op in ops_to_check:
        status = check_ttnn_support(op)
        count = UNET_OPS.get(op, 0)

        if status["exists"]:
            category = status["type"]
            if category not in results:
                category = "direct"
            results[category].append({**status, "count": count})
            symbol = "✅" if status["type"] == "direct" else "⚠️"
        else:
            results["unsupported"].append({**status, "count": count})
            symbol = "❌"

        ttnn_info = f"ttnn.{status['ttnn_op']}"
        note = f" ({status['note']})" if status["note"] else ""
        print(f"{symbol} {op:15} -> {ttnn_info:20} [{status['type']}]{note}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Direct support:    {len(results['direct'])} ops")
    print(f"With workaround:   {len(results['workaround'])} ops")
    print(f"Host-side:         {len(results['host'])} ops")
    print(f"Unsupported:       {len(results['unsupported'])} ops")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": "UNetVGG19",
        "total_ops_checked": len(ops_to_check),
        "supported": {
            op["pytorch_op"]: {
                "ttnn": f"ttnn.{op['ttnn_op']}",
                "count": op["count"],
                "status": "direct",
            }
            for op in results["direct"]
        },
        "workaround": {
            op["pytorch_op"]: {
                "ttnn": f"ttnn.{op['ttnn_op']}",
                "count": op["count"],
                "status": "workaround",
                "note": op["note"],
            }
            for op in results["workaround"]
        },
        "host_side": {
            op["pytorch_op"]: {
                "ttnn": f"ttnn.{op['ttnn_op']}",
                "count": op["count"],
                "status": "host",
                "note": op["note"],
            }
            for op in results["host"]
        },
        "unsupported": {
            op["pytorch_op"]: {
                "ttnn": f"ttnn.{op['ttnn_op']}",
                "count": op["count"],
                "status": "unsupported",
            }
            for op in results["unsupported"]
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
