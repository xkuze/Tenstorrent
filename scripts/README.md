# Scripts

Utility scripts for TT-NN evaluation, profiling, and demonstration.

## Structure

```
scripts/
├── sanity_check.py           # Environment verification
├── demo_ttnn_basics.py       # TT-NN API demonstration
├── check_ttnn_ops.py         # TTNN operation support checker
├── run_tracer.py             # PyTorch model tracer
└── run_visualizer_profiling.py  # Memory/performance profiling
```

## Scripts Overview

### sanity_check.py

Verifies TT-NN environment is properly configured.

```bash
python scripts/sanity_check.py --device_id 3
```

**Checks:**
1. Package imports (torch, ttnn, lightning, torchvision)
2. TTNN version and config
3. Available devices
4. Device connectivity
5. Basic operations (tensor transfer, add, matmul, relu)

**Output:** PASS/FAIL for each check with PCC validation.

---

### demo_ttnn_basics.py

Interactive demonstration of TT-NN concepts for learning.

```bash
# Run all demos
python scripts/demo_ttnn_basics.py --device_id 3

# Run specific section
python scripts/demo_ttnn_basics.py --device_id 3 --section layouts
python scripts/demo_ttnn_basics.py --device_id 3 --section conversion
python scripts/demo_ttnn_basics.py --device_id 3 --section api
```

**Sections:**

| Section | Topics |
|---------|--------|
| `layouts` | TILE vs ROW_MAJOR layout, DRAM vs L1 memory |
| `conversion` | PyTorch to TT-NN patterns (matmul, relu, add, linear) |
| `api` | Device management, data types, reshape, batching, chaining |

---

### check_ttnn_ops.py

Checks which PyTorch operations have TT-NN equivalents.

```bash
# Check UNet operations
python scripts/check_ttnn_ops.py

# Check specific operations
python scripts/check_ttnn_ops.py --ops conv2d relu batch_norm

# Check all known operations
python scripts/check_ttnn_ops.py --all
```

**Output:**
- Direct support (ttnn.op exists)
- Workaround needed (alternative ttnn function)
- Host-side only (must run on CPU)
- Unsupported

**Generates:** `generated/tracer/metadata/unet_ttnn_ops_mapping.json`

---

### run_tracer.py

Traces PyTorch model execution and generates TT-NN code skeleton.

```bash
# Run all phases
python scripts/run_tracer.py --phase all

# Run specific phase
python scripts/run_tracer.py --phase ops       # Extract operations
python scripts/run_tracer.py --phase tests     # Generate unit tests
python scripts/run_tracer.py --phase codegen   # Generate TTNN code
python scripts/run_tracer.py --phase metadata  # Generate JSON metadata
```

**Phases:**

| Phase | Output |
|-------|--------|
| `ops` | `generated/tracer/metadata/unet_ops_list.txt` |
| `tests` | `generated/tracer/tests/test_unet_ops.py` |
| `codegen` | `generated/tracer/codegen/*.py` |
| `metadata` | `generated/tracer/metadata/*.json` |
| `graph` | `generated/tracer/graphs/unet_traced_graph.svg` |

---

### run_visualizer_profiling.py

Generates memory and performance reports for TT-NN Visualizer.

```bash
# Memory profiling
export TTNN_CONFIG_PATH=configs/vis_config.json
python scripts/run_visualizer_profiling.py --mode memory --device_id 3

# Performance profiling
TT_METAL_DEVICE_PROFILER=1 python scripts/run_visualizer_profiling.py \
    --mode performance --device_id 3
```

**Output:**
- Memory: `generated/visualizer/memory_reports/`
- Performance: `generated/visualizer/profiler_logs/`

**Visualizer:**
```bash
uv run ttnn-visualizer  # Opens http://localhost:8000
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Verify environment | `python scripts/sanity_check.py --device_id 3` |
| Learn TT-NN basics | `python scripts/demo_ttnn_basics.py --device_id 3` |
| Check op support | `python scripts/check_ttnn_ops.py` |
| Trace model | `python scripts/run_tracer.py --phase all` |
| Run tracer tests | `pytest generated/tracer/tests/test_unet_ops.py -v` |
| Profile memory | `TTNN_CONFIG_PATH=configs/vis_config.json python scripts/run_visualizer_profiling.py --mode memory` |

## Requirements

- TT-NN installed and configured
- Device access (check with `python -c "import ttnn; print(ttnn.get_device_ids())"`)
- For profiling: `TTNN_CONFIG_PATH` or `TT_METAL_DEVICE_PROFILER` env vars
