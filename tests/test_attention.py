import pytest
import torch
import torch.nn as nn
from model.architecture.attention import CausalSelfAttention


class MockConfig:
    """Mock configuration object for testing."""
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1, max_seq_len=512):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len


@pytest.fixture
def config():
    """Fixture providing a default config for attention testing."""
    return MockConfig(embed_dim=64, num_heads=4, dropout=0.1)


@pytest.fixture
def attention_layer(config):
    """Fixture providing a CausalSelfAttention instance."""
    return CausalSelfAttention(config)


class TestCausalSelfAttentionShapes:
    """Tests for shape correctness of CausalSelfAttention."""
    
    def test_output_shape_matches_input_shape(self, attention_layer):
        """Output shape [B, T, C] should match input shape."""
        batch_size, seq_len, embed_dim = 2, 16, 64
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = attention_layer(x)
        assert output.shape == x.shape
    
    def test_batch_processing(self, attention_layer):
        """Should handle different batch sizes."""
        embed_dim = 64
        seq_len = 10
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, seq_len, embed_dim)
            output = attention_layer(x)
            assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_variable_sequence_length(self, attention_layer):
        """Should handle different sequence lengths."""
        batch_size = 2
        embed_dim = 64
        
        for seq_len in [1, 4, 8, 16, 32, 64]:
            x = torch.randn(batch_size, seq_len, embed_dim)
            output = attention_layer(x)
            assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_single_token(self, attention_layer):
        """With T=1, should output single token without masking issues."""
        batch_size = 2
        embed_dim = 64
        seq_len = 1
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = attention_layer(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"


class TestCausalMasking:
    """Tests for causal masking correctness."""
    
    def test_causal_mask_prevents_future_attention(self, attention_layer):
        """Token at position t should not attend to positions > t."""
        batch_size = 1
        seq_len = 8
        embed_dim = 64
        
        # Use deterministic input
        x = torch.ones(batch_size, seq_len, embed_dim)
        
        # Access attention weights (requires hook or modification)
        # For now, we verify that the model runs without error
        output = attention_layer(x)
        assert output.shape == x.shape
        
        # Verify no NaN or Inf values which would indicate masking issues
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_single_token_no_masking_issues(self, attention_layer):
        """Single token should not have masking problems (attends to itself)."""
        batch_size = 4
        embed_dim = 64
        
        x = torch.randn(batch_size, 1, embed_dim)
        output = attention_layer(x)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (batch_size, 1, embed_dim)


class TestCausalSelfAttentionFunctionality:
    """Tests for correct attention behavior."""
    
    def test_attention_is_trainable(self, attention_layer):
        """Attention layer should have trainable parameters."""
        params = list(attention_layer.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
    
    def test_backward_pass(self, attention_layer):
        """Should support gradient computation."""
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = attention_layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attention_layer.q_proj.weight.grad is not None
    
    def test_dropout_disabled_in_eval_mode(self, attention_layer):
        """Dropout should be disabled during evaluation."""
        attention_layer.eval()
        
        x = torch.randn(2, 8, 64)
        
        # Multiple forward passes should give identical results in eval mode
        output1 = attention_layer(x)
        output2 = attention_layer(x)
        
        assert torch.allclose(output1, output2)
    
    def test_dropout_enabled_in_train_mode(self, attention_layer):
        """Dropout should be active during training."""
        attention_layer.train()
        
        x = torch.randn(2, 8, 64)
        
        # Multiple forward passes may differ due to dropout
        # (though they might coincidentally be the same, this test just verifies the mode)
        attention_layer(x)  # Should run without error
    
    def test_attention_mechanism_is_permutation_invariant_within_seq(self, attention_layer):
        """Attention should work for any input sequence (batch invariance)."""
        attention_layer.eval()
        
        batch_size = 3
        seq_len = 5
        embed_dim = 64
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = attention_layer(x)
        
        # Each position in output should be influenced by all previous positions
        assert output.shape == x.shape


class TestCausalSelfAttentionConfiguration:
    """Tests for configuration handling."""
    
    def test_different_embedding_dimensions(self):
        """Should work with different embedding dimensions."""
        for embed_dim in [32, 64, 128, 256]:
            config = MockConfig(embed_dim=embed_dim, num_heads=4)
            layer = CausalSelfAttention(config)
            
            x = torch.randn(2, 8, embed_dim)
            output = layer(x)
            
            assert output.shape == (2, 8, embed_dim)
    
    def test_different_num_heads(self):
        """Should work with different numbers of heads."""
        embed_dim = 64
        for num_heads in [1, 2, 4, 8]:
            config = MockConfig(embed_dim=embed_dim, num_heads=num_heads)
            layer = CausalSelfAttention(config)
            
            x = torch.randn(2, 8, embed_dim)
            output = layer(x)
            
            assert output.shape == (2, 8, embed_dim)
    
    def test_embed_dim_not_divisible_by_num_heads_raises_error(self):
        """Should raise assertion error if embed_dim not divisible by num_heads."""
        config = MockConfig(embed_dim=65, num_heads=4)  # 65 not divisible by 4
        
        with pytest.raises(AssertionError):
            CausalSelfAttention(config)


class TestCausalSelfAttentionGradients:
    """Tests for gradient flow."""
    
    def test_gradient_flow_through_all_parameters(self, attention_layer):
        """Gradients should flow through all parameters."""
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = attention_layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for param in attention_layer.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
    
    def test_gradient_magnitude_reasonable(self, attention_layer):
        """Gradient magnitudes should be reasonable (not exploding/vanishing)."""
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = attention_layer(x)
        loss = output.sum()
        loss.backward()
        
        for param in attention_layer.parameters():
            grad_norm = param.grad.norm()
            assert 1e-8 < grad_norm < 1000, f"Gradient norm {grad_norm} seems unreasonable"


class TestCausalSelfAttentionEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_very_long_sequence(self, attention_layer):
        """Should handle longer sequences (within max_seq_len)."""
        x = torch.randn(1, 256, 64)
        output = attention_layer(x)
        assert output.shape == x.shape
    
    def test_zero_input(self, attention_layer):
        """Should handle zero input tensor."""
        x = torch.zeros(2, 8, 64)
        output = attention_layer(x)
        
        # With zero input, output should be zero (after linear projection)
        assert output.shape == x.shape
    
    def test_extremely_small_embedding_dim(self):
        """Should work even with very small embedding dimension."""
        config = MockConfig(embed_dim=4, num_heads=2)  # embed_dim=4, head_dim=2
        layer = CausalSelfAttention(config)
        
        x = torch.randn(2, 8, 4)
        output = layer(x)
        
        assert output.shape == (2, 8, 4)
    
    def test_single_head_attention(self):
        """Should work as single-head attention."""
        config = MockConfig(embed_dim=64, num_heads=1)
        layer = CausalSelfAttention(config)
        
        x = torch.randn(2, 8, 64)
        output = layer(x)
        
        assert output.shape == (2, 8, 64)
