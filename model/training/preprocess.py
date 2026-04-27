import os
import numpy as np
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from datasets import load_dataset

from model.architecture.tokenizer import CustomTokenizer


def stream_tokenized_documents(
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
) -> Generator[np.ndarray, None, None]:
    tokenizer = CustomTokenizer()
    ds = load_dataset(dataset_name, split=split, streaming=True) # Streams the data progressively while iterating on the dataset.
    
    print(f"Loading {dataset_name} in streaming mode...")
    
    for example in ds:
        text = example.get("text") or example.get("story", "")
        
        if text:
            tokens = tokenizer.encode_with_endoftext(text)
            yield np.array(tokens, dtype=np.uint16)


def process_dataset(
    output_dir: str = "data",
    train_split: float = 0.9,
    chunk_size: int = 1024 * 1024,  
    dataset_name: str = "roneneldan/TinyStories",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"
    
    tokenizer = CustomTokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"Vocab size: {vocab_size} (fits in uint16: {vocab_size < 65535})", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    print("\nFirst pass: counting total tokens...", flush=True)
    total_tokens = 0
    num_documents = 0
    
    for tokens in stream_tokenized_documents(dataset_name):
        total_tokens += len(tokens)
        num_documents += 1
    
    train_tokens_target = int(total_tokens * train_split)
    
    print(f"Total documents: {num_documents}", flush=True)
    print(f"Total tokens: {total_tokens}", flush=True)
    print(f"Train target: {train_tokens_target} ({train_split*100:.0f}%)", flush=True)
    print(f"Val target: {total_tokens - train_tokens_target} ({(1-train_split)*100:.0f}%)", flush=True)
    
    print("\nSecond pass: writing binary files...", flush=True)
    
    train_tokens_written = 0
    val_tokens_written = 0
    chunk_buffer = np.array([], dtype=np.uint16)
    
    train_mmap = np.memmap(train_bin_path, dtype=np.uint16, mode="w+", shape=(train_tokens_target,))
    val_mmap = np.memmap(val_bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens - train_tokens_target,))
    
    pbar = tqdm(total=total_tokens, unit="tokens", desc="Processing")
    last_update = 0
    
    for tokens in stream_tokenized_documents(dataset_name):
        chunk_buffer = np.concatenate([chunk_buffer, tokens])
        
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
    
    train_mmap.flush()
    val_mmap.flush()
    del train_mmap
    del val_mmap
    
    os.truncate(train_bin_path, train_tokens_written * 2)  # *2 for uint16 (2 bytes)
    os.truncate(val_bin_path, val_tokens_written * 2)  
    
    print(f"\n✓ Training split: {train_bin_path}", flush=True)
    print(f"  Tokens: {train_tokens_written:,}", flush=True)
    print(f"  Size: {train_bin_path.stat().st_size / (1024**3):.2f} GB", flush=True)
    
    print(f"\n✓ Validation split: {val_bin_path}", flush=True)
    print(f"  Tokens: {val_tokens_written:,}", flush=True)
    print(f"  Size: {val_bin_path.stat().st_size / (1024**3):.2f} GB", flush=True)
    
    print(f"\nActual split: {train_tokens_written / (train_tokens_written + val_tokens_written) * 100:.1f}% train", flush=True)


if __name__ == "__main__":
    process_dataset(
        output_dir="data",
        train_split=0.9,
        dataset_name="roneneldan/TinyStories",
    )
