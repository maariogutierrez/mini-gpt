import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .transformer_block import TransformerBlock
from .positional_encoding import LearnedPositionalEmbedding


@dataclass
class GPTConfig:
    """Configuration for GPT model.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        block_size (int): Maximum sequence length (context window)
        n_layer (int): Number of transformer blocks
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension / hidden size
        dropout (float): Dropout rate for regularization
    """
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1


class GPT(nn.Module):
    """GPT-style language model.
    
    A transformer-based autoregressive language model consisting of:
    - Token embeddings
    - Positional embeddings  
    - Stack of transformer blocks with causal self-attention
    - Final layer normalization
    - Output projection to vocabulary
    
    Args:
        config (GPTConfig): Model configuration
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings: vocabulary → embedding dimension
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embeddings: [0, block_size) → embedding dimension
        self.pos_emb = LearnedPositionalEmbedding(config.block_size, config.n_embd)
        
        # Dropout after embeddings
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        # Create a config-like object for TransformerBlock
        self.transformer_blocks = nn.ModuleList([
            self._create_transformer_block() 
            for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.final_ln = nn.LayerNorm(config.n_embd)
        
        # Output projection: embedding dimension → vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Count and print total parameters
        self._print_model_info()
    
    def _create_transformer_block(self):
        """Create a transformer block with a compatible config object."""
        # Create a simple config-like object for TransformerBlock
        class TransformerBlockConfig:
            pass
        
        tb_config = TransformerBlockConfig()
        tb_config.embed_dim = self.config.n_embd
        tb_config.num_heads = self.config.n_head
        tb_config.dropout = self.config.dropout
        tb_config.max_seq_len = self.config.block_size
        
        return TransformerBlock(tb_config)
    
    def _init_weights(self):
        """Initialize weights in GPT-2 style: Normal(0, 0.02)."""
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                # Weight matrices: normal initialization with std=0.02
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name or 'norm' in name:
                # Biases and layer norms: zero initialization
                nn.init.zeros_(param)
    
    def _print_model_info(self):
        """Count and print total number of parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
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
        """Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape [batch_size, seq_len]
                Each element is in range [0, vocab_size)
        
        Returns:
            logits: Logits of shape [batch_size, seq_len, vocab_size]
                Unnormalized probability distribution over vocabulary for each position
        """
        batch_size, seq_len = idx.shape
        
        # Check that sequence length doesn't exceed block size
        assert seq_len <= self.config.block_size, \
            f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
        
        # Token embeddings: [batch, seq_len] → [batch, seq_len, n_embd]
        token_embeddings = self.token_emb(idx)
        
        # Add positional embeddings: [batch, seq_len, n_embd]
        x = self.pos_emb(token_embeddings)
        
        # Apply dropout to embeddings
        x = self.emb_dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Final layer normalization
        x = self.final_ln(x)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Generate new tokens autoregressively.
        
        Starting from initial token indices, generate tokens one at a time
        using the model's predictions, with optional temperature scaling
        and top-k filtering for diversity control.
        
        Args:
            idx: Initial token indices of shape [batch_size, prompt_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for softmax (higher = more random).
                Values < 1 sharpen distribution, > 1 flatten distribution.
                Default: 1.0 (no scaling)
            top_k: If set, only sample from top-k most likely tokens.
                Helps avoid very unlikely tokens. Default: None (no filtering)
        
        Returns:
            Generated token indices of shape [batch_size, prompt_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Get current sequence length
            seq_len = idx.shape[1]
            
            # Crop to block_size if necessary
            idx_cond = idx if seq_len <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass: get logits for next token
            logits = self(idx_cond)  # [batch, seq_len, vocab_size]
            
            # Get logits for the last token (next position to predict)
            logits = logits[:, -1, :]  # [batch, vocab_size]
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering (optional)
            if top_k is not None:
                # Get top-k logits and their indices
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                
                # Create a mask for top-k values
                min_logit = top_k_logits[:, -1:]  # [batch, 1]
                logits = torch.where(
                    logits >= min_logit,
                    logits,
                    torch.full_like(logits, float('-inf'))
                )
            
            # Convert logits to probabilities with softmax
            probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]
            
            # Sample next token from the distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Append to sequence
            idx = torch.cat([idx, next_idx], dim=1)  # [batch, seq_len + 1]
        
        return idx
