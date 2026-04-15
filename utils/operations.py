import torch
import numpy as np

def consensus_degree(opinions_tensor):
    """
    Computes consensus degree. 
    Assumes opinions_tensor is already on the desired device.
    """
    if not isinstance(opinions_tensor, torch.Tensor):
        raise TypeError("consensus_degree expects a torch.Tensor input")

    # 1. Ensure it's a 1D tensor
    if opinions_tensor.ndim != 1:
        opinions_tensor = opinions_tensor.flatten()

    num_opinions = opinions_tensor.numel()
    if num_opinions == 0:
        return 0.0

    # 2. Vectorized calculation
    mean_opinion = torch.mean(opinions_tensor)
    total_deviation = torch.sum(torch.abs(opinions_tensor - mean_opinion))
    consensus = 1.0 - (total_deviation / num_opinions)

    return torch.clamp(consensus, 0.0, 1.0)


def normalize_weights(weights, return_tensor=True):
    """
    Normalizes a weight matrix into a row-stochastic matrix.
    Assumes weights is already a tensor on the correct device.
    """
    if not isinstance(weights, torch.Tensor):
        raise TypeError("normalize_weights expects a torch.Tensor input")

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weight_matrix must be a square 2D matrix")

    row_sums = torch.sum(weights, dim=1, keepdim=True)
    
    # 3. Efficient Normalization
    # Avoid creating a new zero tensor; divide in-place where possible
    normalized = weights / (row_sums + 1e-9) 
    
    # 4. Handle isolated nodes (rows summing to 0)
    isolated_mask = (row_sums.flatten() == 0)
    if torch.any(isolated_mask):
        normalized[isolated_mask, isolated_mask] = 1.0

    if return_tensor:
        return normalized

    return normalized.detach().cpu().numpy()