from __future__ import annotations

import math
import torch


class Linear(torch.nn.Module):
    """
    A Linear layer implementation that performs the transformation y = x @ W.T,
    where W has shape (out_features, in_features).
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.
        This function should accept the following parameters:
        in_features: int
        Final dimension of the input
        out_features: int
        Final dimension of the output
        device: torch.device | None = None
        Device to store the parameters on
        dtype: torch.dtype | None = None
        Data type of the parameters
        """

        super().__init__()
        
        # Create weight param with shape (out_features, in_features)
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        # Initialize with truncated normal distribution
        std = math.sqrt(2.0 / (in_features+out_features))
        a = -3 * std
        b = 3 * std
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return x @ self.W.T
