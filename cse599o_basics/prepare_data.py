import numpy as np
from cse599o_basics.tokenizer import BPETokenizer

# Use your TinyStories files
train_path = "../data/TinyStoriesV2-GPT4-train.txt"
valid_path = "../data/TinyStoriesV2-GPT4-valid.txt"

print("Initializing tokenizer...")
tokenizer = BPETokenizer(vocab=None, merges=None)

print("Encoding training data...")
with open(train_path, "r", encoding="utf-8") as f:
    train_text = f.read()
train_tokens = tokenizer.encode(train_text)

print("Encoding validation data...")
with open(valid_path, "r", encoding="utf-8") as f:
    valid_text = f.read()
valid_tokens = tokenizer.encode(valid_text)

print("Saving memmap datasets...")
train_memmap = np.memmap("../data/train_memmap.dat", dtype=np.int32, mode="w+", shape=(len(train_tokens),))
train_memmap[:] = np.array(train_tokens, dtype=np.int32)
valid_memmap = np.memmap("../data/valid_memmap.dat", dtype=np.int32, mode="w+", shape=(len(valid_tokens),))
valid_memmap[:] = np.array(valid_tokens, dtype=np.int32)

print("✅ Done! Saved:")
print(f"  ../data/train_memmap.dat  ({len(train_tokens)} tokens)")
print(f"  ../data/valid_memmap.dat  ({len(valid_tokens)} tokens)")
