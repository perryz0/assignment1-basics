import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Samples a batch of input+target pairs from the token sequence.
    """
    n = len(x) - context_length
    ix = np.random.randint(0, n, size=(batch_size,))

    # Gather seqs
    x_batch = np.stack([x[i:i + context_length] for i in ix])
    y_batch = np.stack([x[i + 1:i + context_length + 1] for i in ix])

    # Convert to tensors
    x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
    y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

    return x_batch, y_batch
