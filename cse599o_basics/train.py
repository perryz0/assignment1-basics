import argparse
import torch
import numpy as np
from pathlib import Path

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.train_utils import cross_entropy, lr_cosine_schedule, gradient_clipping
from cse599o_basics.adamw import AdamW
from cse599o_basics.data_utils import get_batch
from cse599o_basics.checkpoint import save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=10000)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("-" * 60)

    # Memory efficient data loading
    data = np.memmap(args.data_path, dtype=np.int32, mode="r")
    print(f"Loaded dataset of length {len(data)}")
    print("-" * 60)

    # Model + optimizer setup
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

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume and Path(args.checkpoint_path).exists():
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
        print("-" * 60)

    # Training loop
    model.train()
    for iteration in range(start_iter, args.max_iters):

        # Sample batch
        x, y = get_batch(data, args.batch_size, args.context_length, device)

        # Forward, then backward
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)

        # Update learning rate
        lr_t = lr_cosine_schedule(
            t=iteration,
            alpha_max=args.lr,
            alpha_min=1e-5,
            T_w=args.warmup_steps,
            T_c=args.cosine_steps,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        optimizer.step()

        if iteration % 100 == 0:
            print(f"[Iter {iteration:05d}] Loss: {loss.item():.4f} | LR: {lr_t:.2e}")

        # Checkpointing
        if iteration % 1000 == 0 and iteration > 0:
            save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
            print(f"Checkpoint at iter {iteration}")
            print("-" * 60)

    print("Training done")
    print("-" * 60)


if __name__ == "__main__":
    main()
