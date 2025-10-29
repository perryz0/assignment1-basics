from __future__ import annotations

import torch


class Embedding(torch.nn.Module):
    """
    An embedding layer that maps token IDs to embedding vectors.
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.
        This function should accept the following parameters:
        num_embeddings: int
        Size of the vocabulary
        embedding_dim: int
        Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None
        Device to store the parameters on
        dtype: torch.dtype | None = None
        Data type of the parameters
        """
        super().__init__()
        
        # Create embedding matrix
        self.E = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        # Initialize weights
        torch.nn.init.trunc_normal_(self.E, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.E[token_ids]
