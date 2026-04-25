"""
Preprocess TinyStories dataset: stream, tokenize, and save as binary files.

This script:
1. Streams the TinyStories dataset from HuggingFace
2. Tokenizes each story using GPT-2 tokenizer
3. Inserts <|endoftext|> between documents
4. Saves to binary .bin files using numpy.memmap for memory efficiency
5. Creates train.bin (90%) and val.bin (10%) splits
"""

import os
import numpy as np
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from datasets import load_dataset

from model.architecture.tokenizer import Tokenizer


def stream_tokenized_documents(
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
) -> Generator[np.ndarray, None, None]:
    """
    Stream documents from HuggingFace dataset and tokenize them.
    
    Each document is tokenized with <|endoftext|> appended.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        split: Dataset split to use
        
    Yields:
        Token arrays (uint16) for each document with endoftext token
    """
    tokenizer = Tokenizer()
    ds = load_dataset(dataset_name, split=split, streaming=True)
    
    print(f"Loading {dataset_name} in streaming mode...")
    
    for example in ds:
        # Extract text from the example
        text = example.get("text") or example.get("story", "")
        
        if text:
            # Tokenize with endoftext separator
            tokens = tokenizer.encode_with_endoftext(text)
            yield np.array(tokens, dtype=np.uint16)


def process_dataset(
    output_dir: str = "data",
    train_split: float = 0.9,
    chunk_size: int = 1024 * 1024,  # Process 1M tokens at a time
    dataset_name: str = "roneneldan/TinyStories",
) -> None:
    """
    Process TinyStories dataset and save to binary files.
    
    Creates train.bin and val.bin with the specified split ratio.
    
    Args:
        output_dir: Directory to save .bin files
        train_split: Fraction of data for training (rest for validation)
        chunk_size: Number of tokens to accumulate before writing
        dataset_name: Name of dataset on HuggingFace Hub
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"Vocab size: {vocab_size} (fits in uint16: {vocab_size < 65536})")
    print(f"Output directory: {output_dir}")
    
    # First pass: count total tokens to determine split point
    print("\nFirst pass: counting total tokens...")
    total_tokens = 0
    num_documents = 0
    
    for tokens in stream_tokenized_documents(dataset_name):
        total_tokens += len(tokens)
        num_documents += 1
    
    train_tokens_target = int(total_tokens * train_split)
    
    print(f"Total documents: {num_documents}")
    print(f"Total tokens: {total_tokens}")
    print(f"Train target: {train_tokens_target} ({train_split*100:.0f}%)")
    print(f"Val target: {total_tokens - train_tokens_target} ({(1-train_split)*100:.0f}%)")
    
    # Second pass: accumulate and save
    print("\nSecond pass: writing binary files...")
    
    train_tokens_written = 0
    val_tokens_written = 0
    chunk_buffer = np.array([], dtype=np.uint16)
    
    # Create empty files first
    train_mmap = np.memmap(train_bin_path, dtype=np.uint16, mode="w+", shape=(train_tokens_target,))
    val_mmap = np.memmap(val_bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens - train_tokens_target,))
    
    pbar = tqdm(total=total_tokens, unit="tokens", desc="Processing")
    last_update = 0
    
    for tokens in stream_tokenized_documents(dataset_name):
        chunk_buffer = np.concatenate([chunk_buffer, tokens])
        
        # Write when chunk is full
        if len(chunk_buffer) >= chunk_size:
            remaining_train = train_tokens_target - train_tokens_written
            
            if remaining_train > 0:
                train_chunk = chunk_buffer[:remaining_train]
                train_mmap[train_tokens_written:train_tokens_written + len(train_chunk)] = train_chunk
                train_tokens_written += len(train_chunk)
                chunk_buffer = chunk_buffer[remaining_train:]
            
            if len(chunk_buffer) > 0:
                val_chunk = chunk_buffer
                val_mmap[val_tokens_written:val_tokens_written + len(val_chunk)] = val_chunk
                val_tokens_written += len(val_chunk)
                chunk_buffer = np.array([], dtype=np.uint16)
            
            update_amount = train_tokens_written + val_tokens_written - last_update
            pbar.update(update_amount)
            last_update = train_tokens_written + val_tokens_written
    
    # Handle remaining tokens
    if len(chunk_buffer) > 0:
        remaining_train = train_tokens_target - train_tokens_written
        
        if remaining_train > 0:
            train_chunk = chunk_buffer[:remaining_train]
            train_mmap[train_tokens_written:train_tokens_written + len(train_chunk)] = train_chunk
            train_tokens_written += len(train_chunk)
            chunk_buffer = chunk_buffer[remaining_train:]
        
        if len(chunk_buffer) > 0:
            val_mmap[val_tokens_written:val_tokens_written + len(chunk_buffer)] = chunk_buffer
            val_tokens_written += len(chunk_buffer)
        
        update_amount = train_tokens_written + val_tokens_written - last_update
        pbar.update(update_amount)
    
    pbar.close()
    
    # Flush and close memmap arrays
    train_mmap.flush()
    val_mmap.flush()
    del train_mmap
    del val_mmap
    
    # Truncate files to actual size (remove padding)
    os.truncate(train_bin_path, train_tokens_written * 2)  # *2 for uint16 (2 bytes)
    os.truncate(val_bin_path, val_tokens_written * 2)  # *2 for uint16 (2 bytes)
    
    # Print summary
    print(f"\n✓ Training split: {train_bin_path}")
    print(f"  Tokens: {train_tokens_written:,}")
    print(f"  Size: {train_bin_path.stat().st_size / (1024**3):.2f} GB")
    
    print(f"\n✓ Validation split: {val_bin_path}")
    print(f"  Tokens: {val_tokens_written:,}")
    print(f"  Size: {val_bin_path.stat().st_size / (1024**3):.2f} GB")
    
    print(f"\nActual split: {train_tokens_written / (train_tokens_written + val_tokens_written) * 100:.1f}% train")


if __name__ == "__main__":
    # Default: process TinyStories dataset to ./data/
    process_dataset(
        output_dir="data",
        train_split=0.9,
        dataset_name="roneneldan/TinyStories",
    )
