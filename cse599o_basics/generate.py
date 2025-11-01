import torch
import argparse
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.train_utils import decode
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.checkpoint import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load tokenizer ---
    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=["<|endoftext|>"])

    # --- init model ---
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length
    ).to(device)

    # --- load checkpoint ---
    iteration = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint from {args.checkpoint} (iteration {iteration})")

    # --- tokenize prompt ---
    input_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)

    # --- generate ---
    print(f"Prompt: {args.prompt}\n---")
    model.eval()
    with torch.no_grad():
        output_text = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

    print("\nGenerated text:\n")
    print(output_text)

if __name__ == "__main__":
    main()
