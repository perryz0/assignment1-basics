import torch
import einops
from cse599o_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cse599o_basics.rope import RoPE

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.W_q = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_k = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_v = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        self.W_o = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            torch.nn.init.trunc_normal_(w)

        self.attn = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        Q = einops.einsum(x, self.W_q.T, "b s d, D d -> b s D")
        K = einops.einsum(x, self.W_k.T, "b s d, D d -> b s D")
        V = einops.einsum(x, self.W_v.T, "b s d, D d -> b s D")

        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        if self.rope is not None:
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        Q = einops.rearrange(Q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        K = einops.rearrange(K, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        V = einops.rearrange(V, "b s (h d_k) -> b h s d_k", h=self.num_heads)

        mask = ~torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        out = self.attn(Q, K, V, mask)

        out = einops.rearrange(out, "b h s d_k -> b s (h d_k)")
        return einops.einsum(out, self.W_o.T, "b s d, D d -> b s D")
