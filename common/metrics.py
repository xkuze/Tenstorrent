"""Common metrics for model evaluation."""

import torch


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor (must have same shape as tensor1)

    Returns:
        PCC value between -1 and 1, where 1 means perfect correlation
    """
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    t1_centered = t1 - t1.mean()
    t2_centered = t2 - t2.mean()

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered**2).sum() * (t2_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return (numerator / denominator).item()
