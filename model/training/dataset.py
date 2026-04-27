import torch
import numpy as np
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode='r')
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx : idx + self.block_size])
        y = torch.from_numpy(self.tokens[idx + 1 : idx + self.block_size + 1])
        return x, y
