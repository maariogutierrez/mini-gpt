import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for sequence positions.
    
    Uses nn.Embedding to learn positional representations. These embeddings
    are added directly to token embeddings at the input of the model.
    
    Args:
        block_size (int): Maximum sequence length
        n_embd (int): Embedding dimension
    """
    
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        # Create learnable embedding for positions [0, 1, ..., block_size-1]
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
              Token embeddings to which positional embeddings will be added
              
        Returns:
            Tensor of shape [batch_size, seq_len, n_embd] with positional
            embeddings added to token embeddings
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Create position indices [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # Get positional embeddings for each position
        pos_embeddings = self.pos_emb(positions)  # [seq_len, n_embd]
        
        # Add positional embeddings to input
        # Broadcasting handles batch dimension automatically
        return x + pos_embeddings
