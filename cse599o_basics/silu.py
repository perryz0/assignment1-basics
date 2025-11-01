import torch

class SiLU(torch.nn.Module):
    """
    Constructs a SiLU activation function.

    This function should return the following:
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
