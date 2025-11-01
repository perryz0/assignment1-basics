import argparse
import torch
import numpy as np
from pathlib import Path

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.train_utils import cross_entropy, get_batch, decode
from cse599o_basics.optimizer import AdamW
from cse599o_basics.checkpoint import load_checkpoint
from cse599o_basics.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer LM.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation dataset (.bin or memmap).")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset with memmap (validation only)
    val_data = np.memmap(args.data_path, dtype=np.int32, mode="r")

    # Load tokenizer (you can replace with your custom BPE)
    tokenizer = BPETokenizer()

    # Initialize model
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

    # Dummy optimizer just for checkpoint compatibility
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # Load checkpoint
    if Path(args.checkpoint).exists():
        step = load_checkpoint(args.checkpoint, model, optimizer)
        print(f"✅ Loaded checkpoint from step {step}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.eval()

    # Evaluate validation loss
    with torch.no_grad():
        x, y = get_batch(val_data, args.batch_size, args.context_length, device)
        logits = model(x)
        val_loss = cross_entropy(logits, y).item()

    print(f"\n------------------------------")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {np.exp(val_loss):.2f}")
    print(f"------------------------------\n")

    # Generate text using decode()
    print("Sample Generation:")
    print(f"Prompt: {args.prompt}")
    print(f"------------------------------")
    output = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(output)
    print(f"------------------------------")


if __name__ == "__main__":
    main()
