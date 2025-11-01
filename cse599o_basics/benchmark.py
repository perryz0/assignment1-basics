import torch
import torch.cuda.nvtx as nvtx
import argparse
import numpy as np
from cse599o_basics.transformer_lm import TransformerLM


def benchmark_step(model, x, y=None, backward=False, optimizer=None, device="cuda"):
    """Runs one forward (and optionally backward) pass and returns elapsed ms."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with nvtx.range("forward pass"):
        out = model(x)
    
    if backward:
        with nvtx.range("loss computation"):
            loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        with nvtx.range("backward pass"):
            loss.backward()
        if optimizer is not None:
            with nvtx.range("optimizer step"):
                optimizer.step()

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)  # milliseconds

def make_timer_bucket(bucket):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    def pre_hook(*_):
        torch.cuda.synchronize(); start.record()
    def post_hook(*_):
        end.record(); torch.cuda.synchronize()
        bucket.append(start.elapsed_time(end))
    return pre_hook, post_hook


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--optimizer", action="store_true", help="Include optimizer step (requires --backward)")
    parser.add_argument("--annotate_attention", action="store_true", help="Add detailed NVTX annotations to attention operations")
    parser.add_argument("--profile_memory", action="store_true", help="Enable PyTorch memory profiling")
    parser.add_argument("--memory_snapshot", type=str, default="memory_snapshot.pickle", help="Output file for memory snapshot")
    args = parser.parse_args()
    
    if args.optimizer and not args.backward:
        parser.error("--optimizer requires --backward")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.profile_memory and device != "cuda":
        parser.error("--profile_memory requires CUDA device")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)
    
    # optionally add detailed attention annotations
    if args.annotate_attention:
        from cse599o_basics.scaled_dot_product_attention import ScaledDotProductAttention
        from cse599o_basics.softmax import Softmax
        import einops
        import math
        import types
        
        def annotated_scaled_dot_product_attention(self, Q, K, V, mask=None):
            """Annotated version of scaled dot-product attention with NVTX ranges."""
            d_k = Q.shape[-1]
            
            with nvtx.range("computing attention scores"):
                scores = einops.einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            
            with nvtx.range("computing softmax"):
                attn = self.softmax(scores, dim=-1)
            
            with nvtx.range("final matmul"):
                out = einops.einsum(attn, V, "... q k, ... k d_v -> ... q d_v")
            
            return out
        
        # replace the forward method of sdpa modules
        for m in model.modules():
            if isinstance(m, ScaledDotProductAttention):
                # bind the annotated function as a method to each module instance
                m.forward = types.MethodType(annotated_scaled_dot_product_attention, m)

    # register hooks for attn + ffn
    ATTN_MS = []
    FFN_MS = []
    from cse599o_basics.multihead_self_attention import MultiHeadSelfAttention
    from cse599o_basics.swiglu import SwiGLU
    for m in model.modules():
        if isinstance(m, MultiHeadSelfAttention):
            pre, post = make_timer_bucket(ATTN_MS)
            # Use default args to capture pre/post in closure
            m.register_forward_pre_hook(lambda *a, pre_fn=pre, **k: pre_fn())
            m.register_forward_hook(lambda *a, post_fn=post, **k: post_fn())
        elif isinstance(m, SwiGLU):
            pre, post = make_timer_bucket(FFN_MS)
            # Use default args to capture pre/post in closure
            m.register_forward_pre_hook(lambda *a, pre_fn=pre, **k: pre_fn())
            m.register_forward_hook(lambda *a, post_fn=post, **k: post_fn())

    model.train()
    # randomly initialize x and y
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    # optimizer if needed
    optimizer = None
    if args.optimizer:
        from cse599o_basics.adamw import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-3)

    # warmup (marked with NVTX so it can be filtered out in nsys)
    print(f"Warming up for {args.warmup_steps} steps...")
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            _ = benchmark_step(model, x, y, backward=args.backward, optimizer=optimizer, device=device)
            if optimizer is not None:
                optimizer.zero_grad()

    # Start memory profiling if requested
    if args.profile_memory:
        print("Starting memory profiling...")
        torch.cuda.empty_cache()  # Clear cache before starting
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    # measure (marked with NVTX for profiling)
    print(f"Measuring {args.measure_steps} steps...")
    times = []
    with nvtx.range("measurement"):
        for i in range(args.measure_steps):
            torch.cuda.synchronize()
            t = benchmark_step(model, x, y, backward=args.backward, optimizer=optimizer, device=device)
            times.append(t)
            print(f"Run {i+1:02d}: {t:.2f} ms")
            if optimizer is not None:
                optimizer.zero_grad()

    # Stop memory profiling and save snapshot
    if args.profile_memory:
        print("Stopping memory profiling and saving snapshot...")
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        print(f"Peak memory usage: {peak_memory:.2f} GB")
        print(f"Memory snapshot saved to: {args.memory_snapshot}")

    avg = np.mean(times)
    std = np.std(times)
    mode = "forward+backward" if args.backward else "forward-only"
    print(f"\nMode: {mode}")
    print(f"Average time: {avg:.2f} ms +- {std:.2f} ms over {args.measure_steps} runs")

    # measure attn + ffn
    if ATTN_MS and FFN_MS:
        attn_avg = sum(ATTN_MS) / len(ATTN_MS)
        ffn_avg  = sum(FFN_MS) / len(FFN_MS)
        total = attn_avg + ffn_avg
        print(f"\nBreakdown (forward-only, averaged over hooks):")
        print(f"  Self-attention: {attn_avg:.2f} ms  ({attn_avg/total*100:.1f} percent)")
        print(f"  FFN:           {ffn_avg:.2f} ms  ({ffn_avg/total*100:.1f} percent)")


if __name__ == "__main__":
    main()
