import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0, \
            "embed_dim must be divisible by num_heads"
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.dropout_rate = config.dropout

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        
        max_seq_len = getattr(config, 'max_seq_len', 2048)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)) == 1
        self.register_buffer("causal_mask", causal_mask)
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # batch_size, seq_len, embed_dim
        
        # [B, T, num_heads, head_dim]
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        
        # [B, num_heads, T, head_dim] 
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        causal_mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float('-inf'))
        
        # softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0) 
        
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        
        # [B, num_heads, T, head_dim] -> [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, C)
        
        output = self.out_proj(attn_output)
        
        return output