import torch
from typing import Optional
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.multihead_self_attention import MultiHeadSelfAttention
from cse599o_basics.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, *, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))
