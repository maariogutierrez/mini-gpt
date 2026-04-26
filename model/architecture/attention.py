import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer for autoregressive models.
    
    Implements scaled dot-product attention with causal masking to prevent
    the model from attending to future tokens during training and inference.
    
    Args:
        config: Configuration object with:
            - embed_dim (int): Total embedding dimension
            - num_heads (int): Number of attention heads
            - dropout (float): Dropout rate for attention weights
            - max_seq_len (int): Maximum sequence length for causal mask cache
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout_rate = config.dropout
        
        # Linear projections for Q, K, V (combined for efficiency)
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Dropout on attention weights
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # Register causal mask as buffer (not a parameter)
        # Shape: [max_seq_len, max_seq_len]
        max_seq_len = getattr(config, 'max_seq_len', 2048)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)) == 1
        self.register_buffer("causal_mask", causal_mask)
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x, attention_mask=None):
        """Forward pass of causal self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask (e.g., for padding)
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        B, T, C = x.shape  # batch_size, seq_len, embed_dim
        
        # Linear projections and reshape for multi-head attention
        # [B, T, C] -> [B, T, C] -> [B, T, num_heads, head_dim]
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        
        # Transpose to [B, num_heads, T, head_dim] for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Apply causal mask: prevent attending to future tokens
        # Future positions are set to -inf so softmax makes them 0
        causal_mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply optional attention mask (e.g., for padding)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)  # Handle NaN from -inf
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        
        # Concatenate heads: [B, num_heads, T, head_dim] -> [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, C)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output
