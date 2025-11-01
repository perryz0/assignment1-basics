import torch
import einops

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        Args:
        theta: Theta value for RoPE.
        d_k: Dimension of query/key vectors (should be even).
        max_seq_len: Maximum sequence length that will be inputted.
        device: torch.device | None. Device to store the buffers on.
        """
        super().__init__()

        # Frequencies for each pair of dims
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        # Position indices
        pos = torch.arange(max_seq_len, device=device)
        # Outer product, (max_seq_len, d_k/2)
        angles = torch.outer(pos, freqs)

        # Precomp cos and sine
        cos = torch.cos(angles)[None, :, :]   # (1, max_seq_len, d_k/2)
        sin = torch.sin(angles)[None, :, :]   # (1, max_seq_len, d_k/2)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.
        Notes:
        - Accept x with an arbitrary number of batch dimensions.
        - token_positions has shape (..., seq_len) and gives absolute
        positions per token along the sequence dimension.
        - Use token_positions to slice (precomputed) cos/sin tensors
        along the sequence dimension.
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        # Stack and rotate 2x2
        x_rot = torch.stack(
            [x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos],
            dim=-1
        )
        return einops.rearrange(x_rot, "... d r -> ... (d r)")

     
