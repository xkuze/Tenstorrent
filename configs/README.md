# Configs

Configuration files for TT-NN profiling and visualization.

## Structure

```
configs/
└── vis_config.json    # TT-NN Visualizer configuration
```

## vis_config.json

Configuration for TT-NN memory profiling and report generation.

```json
{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "unet_memory_report",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `enable_fast_runtime_mode` | bool | Disable for profiling (reduces overhead skipping) |
| `enable_logging` | bool | Enable operation logging |
| `report_name` | string | Name prefix for generated reports |
| `enable_graph_report` | bool | Generate computation graph visualization |
| `enable_detailed_buffer_report` | bool | Track memory buffer allocations |
| `enable_detailed_tensor_report` | bool | Track tensor metadata |
| `enable_comparison_mode` | bool | Compare TT-NN vs PyTorch outputs |

## Usage

### Memory Profiling

```bash
export TTNN_CONFIG_PATH=configs/vis_config.json
python scripts/run_visualizer_profiling.py --mode memory --device_id 3
```

Reports saved to: `generated/visualizer/memory_reports/`

### Performance Profiling

Performance profiling uses environment variable instead of config file:

```bash
TT_METAL_DEVICE_PROFILER=1 python scripts/run_visualizer_profiling.py \
    --mode performance --device_id 3
```

Reports saved to: `generated/visualizer/profiler_logs/`

### Viewing Reports

```bash
uv run ttnn-visualizer
# Open http://localhost:8000
# Load reports from generated/visualizer/
```

## Creating Custom Configs

For different profiling scenarios, create new config files:

```json
{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "custom_report",
    "enable_graph_report": true,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": true,
    "enable_comparison_mode": false
}
```

Usage:
```bash
export TTNN_CONFIG_PATH=configs/custom_config.json
python your_script.py
```
