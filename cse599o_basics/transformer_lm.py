import torch
from cse599o_basics.rope import RoPE
from cse599o_basics.embedding import Embedding
from cse599o_basics.transformer_block import TransformerBlock
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.linear import Linear

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Precompute RoPE used by all layers
        rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)

        # Embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.tok_emb = self.token_embeddings

        # Transformer stack
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.blocks = self.layers

        # Final norm and output head
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm_final = self.ln_final
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.output = self.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len)
        returns: (batch, seq_len, vocab_size)
        """
        # Embed tokens
        h = self.token_embeddings(x)

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Normalize + output projection
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits
