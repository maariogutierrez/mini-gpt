import torch.nn as nn
from .attention import CausalSelfAttention


class FeedForwardNetwork(nn.Module):  
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        
        ffn_hidden_dim = getattr(config, 'ffn_hidden_dim', 4 * config.embed_dim)
        
        # embed_dim → hidden_dim
        self.fc1 = nn.Linear(config.embed_dim, ffn_hidden_dim)
        
        self.gelu = nn.GELU()
        
        # hidden_dim → embed_dim
        self.fc2 = nn.Linear(ffn_hidden_dim, config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # Linear → GELU → Linear → Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        
        self.attention = CausalSelfAttention(config)
        
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.ffn = FeedForwardNetwork(config)
    
    def forward(self, x):
        # x = x + Attention(LayerNorm(x))
        x = x + self.attention(self.norm1(x))
        
        # x = x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        
        return x