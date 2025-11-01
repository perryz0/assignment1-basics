import torch

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # subtract max for numerical stability
        x_max = torch.amax(x, dim=dim, keepdim=True)
        x_exp = torch.exp(x - x_max)
        x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        return x_exp / x_sum