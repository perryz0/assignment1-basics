import torch
import einops
from typing import Optional
from cse599o_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cse599o_basics.softmax import Softmax
from cse599o_basics.rope import RoPE

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Optional[RoPE] = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.rope = rope
        self.sdpa = ScaledDotProductAttention()

        # Weights shaped to project all heads in one matmul per Q/K/V
        # w_q/w_k/w_v: (h*d_k, d_model), w_o: (d_model, h*d_v)
        self.w_q = torch.nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_k = torch.nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_v = torch.nn.Parameter(torch.empty((num_heads * self.d_v, d_model), device=device, dtype=dtype))
        self.w_o = torch.nn.Parameter(torch.empty((d_model, num_heads * self.d_v), device=device, dtype=dtype))

        for w in (self.w_q, self.w_k, self.w_v, self.w_o):
            torch.nn.init.trunc_normal_(w)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (..., seq, d_model)
        token_positions: (..., seq) or None (RoPE applied only if provided)
        returns: (..., seq, d_model)
        """
        seq = x.shape[-2]

        # Reshape weights per-head for batched projections
        # (h*d_k, d_model) -> (h, d_k, d_model)
        w_q = einops.rearrange(self.w_q, '(h d_k) d_model -> h d_k d_model', h=self.num_heads)
        w_k = einops.rearrange(self.w_k, '(h d_k) d_model -> h d_k d_model', h=self.num_heads)
        w_v = einops.rearrange(self.w_v, '(h d_v) d_model -> h d_v d_model', h=self.num_heads)

        # Project Q/K/V: (..., h, seq, d_k/d_v)
        q = einops.einsum(x, w_q, '... s d_model, h d_k d_model -> ... h s d_k')
        k = einops.einsum(x, w_k, '... s d_model, h d_k d_model -> ... h s d_k')
        v = einops.einsum(x, w_v, '... s d_model, h d_v d_model -> ... h s d_v')

        # Apply RoPE to Q and K if provided
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal mask: (seq, seq), broadcastable to (..., h, seq, seq)
        mask = ~torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=x.device), diagonal=1)

        # Scaled dotprod attention
        attn = self.sdpa(q, k, v, mask)

        # Concat heads and then output projection
        concat = einops.rearrange(attn, '... h s d_v -> ... s (h d_v)')           # (..., seq, h*d_v)
        out = einops.einsum(concat, self.w_o, '... s h_d_v, d_model h_d_v -> ... s d_model')
        return out
