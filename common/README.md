# Common

Shared utilities across all tasks.

## Structure

```
common/
├── __init__.py
└── metrics.py    # compute_pcc
```

## Usage

```python
from common.metrics import compute_pcc

pcc = compute_pcc(pytorch_output, ttnn_output)
```

## Metrics

**compute_pcc** — Pearson Correlation Coefficient between two tensors. Returns value between -1 and 1. Used to verify TT-NN outputs match PyTorch reference. Target: PCC > 0.99.
