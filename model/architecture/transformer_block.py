import torch
import torch.nn as nn
from .attention import CausalSelfAttention


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network component of a Transformer block.
    
    Implements the standard two-layer feed-forward network with GELU activation:
    Linear(embed_dim → hidden_dim) → GELU → Linear(hidden_dim → embed_dim) → Dropout
    
    Args:
        config: Configuration object with:
            - embed_dim (int): Embedding dimension
            - dropout (float): Dropout rate
            - ffn_hidden_dim (int, optional): Hidden dimension. If not provided,
              defaults to 4 × embed_dim (standard transformer scaling)
    """
    
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        
        # Hidden dimension defaults to 4× embedding dimension (standard practice)
        ffn_hidden_dim = getattr(config, 'ffn_hidden_dim', 4 * config.embed_dim)
        
        # First linear layer: embed_dim → hidden_dim
        self.fc1 = nn.Linear(config.embed_dim, ffn_hidden_dim)
        
        # GELU activation function
        self.gelu = nn.GELU()
        
        # Second linear layer: hidden_dim → embed_dim
        self.fc2 = nn.Linear(ffn_hidden_dim, config.embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Linear → GELU → Linear → Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm residual connections.
    
    Implements the transformer building block:
    x = x + Attention(LayerNorm(x))      # with residual connection
    x = x + FFN(LayerNorm(x))            # with residual connection
    
    Args:
        config: Configuration object with:
            - embed_dim (int): Embedding dimension
            - num_heads (int): Number of attention heads
            - dropout (float): Dropout rate
            - max_seq_len (int): Maximum sequence length for causal mask
            - ffn_hidden_dim (int, optional): FFN hidden dimension
    """
    
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        
        # Pre-norm: LayerNorm before attention
        self.norm1 = nn.LayerNorm(config.embed_dim)
        
        # Multi-head self-attention
        self.attention = CausalSelfAttention(config)
        
        # Pre-norm: LayerNorm before feed-forward
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(config)
    
    def forward(self, x):
        """Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Pre-norm residual connection for attention
        # x = x + Attention(LayerNorm(x))
        x = x + self.attention(self.norm1(x))
        
        # Pre-norm residual connection for feed-forward network
        # x = x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        
        return x
