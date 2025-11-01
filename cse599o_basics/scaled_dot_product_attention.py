import torch
import einops
import math
from cse599o_basics.softmax import Softmax  # import your own module

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor (..., seq_len_q, d_k)
            K: Key tensor (..., seq_len_k, d_k)
            V: Value tensor (..., seq_len_k, d_v)
            mask: Optional bool tensor (..., seq_len_q, seq_len_k)
                where True = attend, False = mask out.

        Returns:
            torch.Tensor: (..., seq_len_q, d_v)
        """
        d_k = Q.shape[-1]
        scores = einops.einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = self.softmax(scores, dim=-1)
        out = einops.einsum(attn, V, "... q k, ... k d_v -> ... q d_v")
        return out
