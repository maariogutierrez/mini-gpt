import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        pos_embeddings = self.pos_emb(positions)  # [seq_len, n_embd]
        
        return x + pos_embeddings