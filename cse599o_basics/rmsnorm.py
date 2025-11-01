import torch
import einops

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        This function should accept the following parameters:
        
        d_model: int # Hidden dimension of the model
        eps: float = 1e-5 # Epsilon value for numerical stability
        device: torch.device | None # Device to store the parameters on
        dtype: torch.dtype | None # Data type of the parameters
        """

        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""

        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einops.reduce(x ** 2, "b s d -> b s 1", "mean") + self.eps)

        result = (x / rms) * self.weight
        return result.to(in_dtype)
