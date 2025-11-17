# benchmark.py (ran on Tomago)
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utility: run a forward-backward step with optional checkpointing
# ---------------------------------------------------------------------

def _call_block(block, hidden_states):
    """Helper to call block and extract tensor from tuple output."""
    output = block(hidden_states)
    return output[0] if isinstance(output, tuple) else output

def run_step(model, input_ids, use_cp=False, boundaries=None):
    """
    boundaries: list of layer indices to checkpoint OR None
    if boundaries is None and use_cp=True, we instead do uniform checkpointing
    """
    # 1. Embed tokens (GPT-2 does this in forward())
    hidden_states = model.transformer.wte(input_ids)
    hidden_states = model.transformer.drop(hidden_states)

    # 2. Iterate over transformer blocks
    for i, block in enumerate(model.transformer.h):
        if use_cp:
            if boundaries is not None and i in boundaries:
                # boundary-aware checkpointing
                hidden_states = checkpoint(_call_block, block, hidden_states)
            elif boundaries is None:
                # uniform checkpointing: checkpoint every layer
                hidden_states = checkpoint(_call_block, block, hidden_states)
            else:
                hidden_states = _call_block(block, hidden_states)
        else:
            hidden_states = _call_block(block, hidden_states)

    # 3. Final layers
    hidden_states = model.transformer.ln_f(hidden_states)
    logits = model.lm_head(hidden_states)

    loss = logits.mean()
    loss.backward()
    return loss.item()


# ---------------------------------------------------------------------
# Simple heuristics
# ---------------------------------------------------------------------

def uniform_boundaries(num_layers):
    """Checkpoint every layer."""
    return None  # None = checkpoint everything (uniform)


def boundary_aware(num_layers, shard_size=4):
    """
    Fake 'FSDP boundary aware': avoid checkpointing inside shard groups.
    Simulate shard groups of size shard_size.
    
    Example: if shard_size=4 and num_layers=12 → shards = [0-3], [4-7], [8-11]
    We checkpoint only at edges (3, 7, 11).
    """
    boundaries = []
    for i in range(shard_size - 1, num_layers, shard_size):
        boundaries.append(i)
    return boundaries

# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def benchmark(model, input_ids, use_uniform=False, use_boundary=False):
    torch.cuda.empty_cache()

    steps = 8
    times = []

    # warmup step (forward only, no backward)
    with torch.no_grad():
        for _ in range(2):
            hidden_states = model.transformer.wte(input_ids)
            hidden_states = model.transformer.drop(hidden_states)
            for block in model.transformer.h:
                output = block(hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states = model.transformer.ln_f(hidden_states)
            _ = model.lm_head(hidden_states)

    for _ in range(steps):
        model.zero_grad(set_to_none=True)

        if use_uniform:
            t0 = time.time()
            run_step(model, input_ids, use_cp=True, boundaries=None)
            t1 = time.time()

        elif use_boundary:
            num_layers = len(model.transformer.h)
            b = boundary_aware(num_layers)
            t0 = time.time()
            run_step(model, input_ids, use_cp=True, boundaries=b)
            t1 = time.time()

        else:
            t0 = time.time()
            run_step(model, input_ids, use_cp=False)
            t1 = time.time()

        times.append(t1 - t0)

    return sum(times) / len(times)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # small GPT-2 config
    cfg = GPT2Config(
        n_embd=256,
        n_layer=12,
        n_head=4,
        n_positions=128,
        vocab_size=50000,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()  # Ensure model is in training mode

    # random input
    B = 2
    T = 128
    input_ids = torch.randint(0, 50000, (B, T)).to(device)

    print("\nRunning benchmarks...")

    t_no_cp = benchmark(model, input_ids, use_uniform=False, use_boundary=False)
    print(f"No checkpointing:       {t_no_cp:.4f} sec")

    t_uniform = benchmark(model, input_ids, use_uniform=True, use_boundary=False)
    print(f"Uniform checkpointing:  {t_uniform:.4f} sec")

    t_boundary = benchmark(model, input_ids, use_uniform=False, use_boundary=True)
    print(f"Boundary-aware CP:      {t_boundary:.4f} sec")

    # plot results
    labels = ["No-CP", "Uniform-CP", "Boundary-CP"]
    values = [t_no_cp, t_uniform, t_boundary]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.ylabel("Time per step (sec)")
    plt.title("Checkpointing Benchmark")
    plt.savefig("benchmark_results.png")

    print("\nSaved plot to benchmark_results.png")


if __name__ == "__main__":
    main()
