import torch
import einops
from cse599o_basics.silu import SiLU

class SwiGLU(torch.nn.Module):
    """
    SwiGLU feedforward: SiLU(x @ W1.T) * (x @ W3.T), projected back with W2.

    Args:
        d_model (int): The dimension of the model.
        d_ff (int | None): The dimension of the feedforward layer. If None, it is set to 8/3 * d_model.
        device (torch.device | None): The device to store the parameters on.
        dtype (torch.dtype | None): The data type of the parameters.
        
    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        # Calculate d_ff and fix dimensionality
        d_ff = int((8/3) * d_model)
        d_ff = d_ff - (d_ff % 64)

        # Initialize weights
        self.W1 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2 = torch.nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.W3 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W1)
        torch.nn.init.trunc_normal_(self.W2)
        torch.nn.init.trunc_normal_(self.W3)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        w1x = einops.einsum(x, self.W1, "b s d, f d -> b s f")
        w3x = einops.einsum(x, self.W3, "b s d, f d -> b s f")
        glu = self.silu(w1x) * w3x
        out = einops.einsum(glu, self.W2, "b s f, d f -> b s d")
        return out
