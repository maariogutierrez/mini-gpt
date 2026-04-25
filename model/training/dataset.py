import numpy as np
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, bin_path, block_size):
        """
        Initialize TokenDataset with a memory-mapped binary file.
        
        Args:
            bin_path (str): Path to the .bin file containing tokens
            block_size (int): Context window size for predictions
        """
        self.block_size = block_size
        # Load tokens as numpy memmap (never loads whole file into memory)
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
    
    def __len__(self):
        """Return the number of available samples."""
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        """
        Get a training sample at the given index.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (x, y) where
                x = tokens[idx : idx + block_size] (input context)
                y = tokens[idx+1 : idx + block_size + 1] (next-token targets)
        """
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y
