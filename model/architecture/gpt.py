import torch
import torch.nn as nn
from dataclasses import dataclass
from types import SimpleNamespace
from .transformer_block import TransformerBlock
from .positional_encoding import LearnedPositionalEmbedding


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1


class GPT(nn.Module):  
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # vocabulary → embedding dimension
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # [0, block_size) → embedding dimension
        self.pos_emb = LearnedPositionalEmbedding(config.block_size, config.n_embd)
        
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(SimpleNamespace(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                dropout=config.dropout,
                max_seq_len=config.block_size
            )) for _ in range(config.n_layer)
        ])
        
        self.final_ln = nn.LayerNorm(config.n_embd)
        
        # embedding dimension → vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.token_emb.weight = self.lm_head.weight
        
        self._init_weights()
        
        self._print_model_info()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'token_emb' in name:
                continue
            if 'norm' in name:
                continue
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name or 'norm' in name:
                nn.init.zeros_(param)
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in set(self.parameters()))
        trainable_params = sum(p.numel() for p in set(self.parameters()) if p.requires_grad)
        
        print(f"GPT Model Information:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Config:")
        print(f"    - vocab_size: {self.config.vocab_size}")
        print(f"    - block_size: {self.config.block_size}")
        print(f"    - n_layer: {self.config.n_layer}")
        print(f"    - n_head: {self.config.n_head}")
        print(f"    - n_embd: {self.config.n_embd}")
        print(f"    - dropout: {self.config.dropout}")
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        
        assert seq_len <= self.config.block_size, \
            f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
        
        # [batch, seq_len] → [batch, seq_len, n_embd]
        token_embeddings = self.token_emb(idx)
        
        # [batch, seq_len, n_embd]
        x = self.pos_emb(token_embeddings)
        
        x = self.emb_dropout(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.final_ln(x)
        
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            seq_len = idx.shape[1]
            
            idx_cond = idx if seq_len <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits = self(idx_cond)  # [batch, seq_len, vocab_size]
            
            logits = logits[:, -1, :]  # [batch, vocab_size]
            
            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx
