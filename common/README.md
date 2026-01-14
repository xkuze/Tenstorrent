# Common

Shared utilities and metrics across all model tasks.

## Structure

```
common/
├── __init__.py
└── metrics.py    # PCC and other metrics
```

## Metrics

### compute_pcc

Computes Pearson Correlation Coefficient between two tensors.

```python
from common.metrics import compute_pcc

# Compare PyTorch and TT-NN outputs
pcc = compute_pcc(pytorch_output, ttnn_output)
print(f"PCC: {pcc:.6f}")  # e.g., PCC: 0.999985
```

**Parameters:**
- `tensor1`: First tensor (any shape)
- `tensor2`: Second tensor (same shape as tensor1)

**Returns:**
- Float value between -1 and 1
- 1.0 = perfect positive correlation
- 0.0 = no correlation
- -1.0 = perfect negative correlation

**Formula:**
```
PCC = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)
```

## Usage in Project

PCC is used to validate TT-NN inference matches PyTorch reference:

```python
# In inference_ttnn.py
pytorch_output = run_inference_pytorch(model, images)
ttnn_output = run_inference_ttnn(model, images, device)

pcc = compute_pcc(pytorch_output, ttnn_output)
print(f"PCC > 0.99: {'YES' if pcc > 0.99 else 'NO'}")
```

**Target:** PCC > 0.99 indicates successful TT-NN implementation.

## Thresholds

| PCC Value | Interpretation |
|-----------|----------------|
| > 0.999 | Excellent match (bfloat16 precision expected) |
| > 0.99 | Good match (acceptable for inference) |
| > 0.95 | Moderate match (may need investigation) |
| < 0.95 | Poor match (likely implementation issue) |

## Adding New Metrics

To add new metrics, edit `common/metrics.py`:

```python
def compute_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Mean Squared Error."""
    return ((tensor1 - tensor2) ** 2).mean().item()

def compute_mae(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    return (tensor1 - tensor2).abs().mean().item()
```

Then update `__init__.py` to export:

```python
from common.metrics import compute_pcc, compute_mse, compute_mae
```
