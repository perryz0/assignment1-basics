import torch
import einops
import math
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.softmax import Softmax

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean cross-entropy loss over all examples and positions.

    Args:
        logits: (..., vocab_size)
        targets: (...) same batch/seq shape without vocab dimension

    Returns:
        scalar loss (mean)
    """

    logits = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    correct = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    loss = -(correct - logsumexp)
    return loss.mean()

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Cosine learning rate schedule with warm-up.
    """
    # Warm up
    if t < T_w:
        return (t / T_w) * alpha_max

    # Cosine annealing
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_term * (alpha_max - alpha_min)

    # Post-annealing
    else:
        return alpha_min
    
def gradient_clipping(params, max_norm: float, eps: float = 1e-6):
    """
    Gradient clipping by L2 norm.
    """
    # Flatten
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0

    # Find total L2 norm of all gradients
    total_norm = torch.sqrt(sum((p.grad.data.norm(2) ** 2 for p in params)))

    # Compute clipping factor, 1 if below max_norm
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params:
            p.grad.data.mul_(clip_coef)

    return total_norm

@torch.no_grad()
def decode(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Generate text from a trained Transformer language model using temperature
    scaling and top-p (nucleus) sampling.

    Args:
        model: Trained TransformerLM.
        tokenizer: Tokenizer with encode() / decode() methods.
        prompt: Text prompt to start generation.
        max_tokens: Maximum number of tokens to generate (including prompt).
        temperature: Sampling temperature (τ).
        top_p: Nucleus sampling threshold.
        device: Device to run inference on.
    """

    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = tokens.tolist()[0]

    for _ in range(max_tokens - len(generated)):
        logits = model(tokens)                     # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :] / temperature
        probs = Softmax()(next_logits, dim=-1)

        # top-p (nucleus) sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        mask = cumulative <= top_p
        mask[torch.sum(mask)] = True  # include the first above threshold
        filtered_probs = sorted_probs[mask]
        filtered_idx = sorted_idx[mask]
        filtered_probs /= filtered_probs.sum()

        next_token = filtered_idx[torch.multinomial(filtered_probs, 1)]
        token_id = next_token.item()
        generated.append(token_id)

        # stop if we hit the <|endoftext|> token id
        if token_id == tokenizer.encode("<|endoftext|>")[0]:
            break

        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

    return tokenizer.decode(generated)
