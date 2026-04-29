import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from model.architecture.tokenizer import CustomTokenizer

def process_dataset(
    output_dir: str = "data",
    train_split: float = 0.9,
    dataset_name: str = "roneneldan/TinyStories",
    seed: int = 42
) -> None:

    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"
    
    tokenizer = CustomTokenizer()
    ds = load_dataset(dataset_name, split="train", streaming=True) # Streams the data progressively while iterating on the dataset.
    
    print(f"Starting processing...")
    print(f"Vocab size: {tokenizer.vocab_size}")

    with open(train_bin_path, "wb") as f_train, open(val_bin_path, "wb") as f_val:
        train_tokens = 0
        val_tokens = 0
        
        pbar = tqdm(unit="docs", desc="Processing Documents")
        
        for example in ds:
            text = example.get("text") or example.get("story", "")
            if not text:
                continue
            
            tokens = tokenizer.encode_with_endoftext(text)
            token_arr = np.array(tokens, dtype=np.uint16)
            
            if random.random() < train_split:
                f_train.write(token_arr.tobytes())
                train_tokens += len(tokens)
            else:
                f_val.write(token_arr.tobytes())
                val_tokens += len(tokens)
            
            pbar.update(1)
        
        pbar.close()

    total_tokens = train_tokens + val_tokens
    print(f"\n✓ Processing Complete!")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Train: {train_tokens:,} tokens ({(train_tokens/total_tokens)*100:.1f}%)")
    print(f"Val:   {val_tokens:,} tokens ({(val_tokens/total_tokens)*100:.1f}%)")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    process_dataset(
        output_dir="data",
        train_split=0.9,
        dataset_name="roneneldan/TinyStories",
    )
