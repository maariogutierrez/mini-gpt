import pytest
from model.architecture.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    """Fixture providing a Tokenizer instance."""
    return Tokenizer()


class TestTokenizerBasic:
    """Tests for basic tokenizer functionality."""
    
    def test_encode_returns_list_of_ints(self, tokenizer):
        """Encode should return a list of integers."""
        tokens = tokenizer.encode("hello")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_decode_returns_string(self, tokenizer):
        """Decode should return a string."""
        tokens = tokenizer.encode("hello")
        text = tokenizer.decode(tokens)
        assert isinstance(text, str)
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Encoding then decoding should recover the original string."""
        original = "Hello, world! This is a test of the tokenizer."
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original
    
    def test_encode_decode_roundtrip_with_special_chars(self, tokenizer):
        """Roundtrip should work with special characters."""
        original = "🚀 Special chars: !@#$%^&*() \n\t quotes: \"hello\""
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original
    
    def test_encode_empty_string(self, tokenizer):
        """Encoding empty string should return empty list."""
        tokens = tokenizer.encode("")
        assert tokens == []
    
    def test_decode_empty_list(self, tokenizer):
        """Decoding empty list should return empty string."""
        text = tokenizer.decode([])
        assert text == ""
    
    def test_vocab_size_is_positive_int(self, tokenizer):
        """Vocabulary size should be a positive integer."""
        vocab = tokenizer.vocab_size
        assert isinstance(vocab, int)
        assert vocab > 0


class TestTokenizerKnownStrings:
    """Tests for known string to token mappings."""
    
    def test_space_token(self, tokenizer):
        """Single space should encode consistently."""
        space_tokens = tokenizer.encode(" ")
        assert len(space_tokens) == 1
        
        # Verify it decodes back
        assert tokenizer.decode(space_tokens) == " "
    
    def test_number_encoding(self, tokenizer):
        """Numbers should encode to consistent tokens."""
        num_str = "12345"
        tokens1 = tokenizer.encode(num_str)
        tokens2 = tokenizer.encode(num_str)
        assert tokens1 == tokens2
    
    def test_same_text_same_tokens(self, tokenizer):
        """Same text should always produce same tokens."""
        text = "consistent encoding test"
        tokens1 = tokenizer.encode(text)
        tokens2 = tokenizer.encode(text)
        tokens3 = tokenizer.encode(text)
        assert tokens1 == tokens2 == tokens3
    
    def test_single_character_encoding(self, tokenizer):
        """Single characters should encode consistently."""
        for char in "abcABC123!@#":
            tokens1 = tokenizer.encode(char)
            tokens2 = tokenizer.encode(char)
            assert tokens1 == tokens2
            assert len(tokens1) > 0


class TestTokenizerSpecialTokens:
    """Tests for special token handling."""
    
    def test_endoftext_token_exists(self, tokenizer):
        """Endoftext token should be encodable."""
        endoftext_id = tokenizer._endoftext_id
        assert isinstance(endoftext_id, int)
        assert endoftext_id > 0
    
    def test_encode_with_endoftext(self, tokenizer):
        """Encoding with endoftext should append the special token."""
        text = "sample text"
        tokens_normal = tokenizer.encode(text)
        tokens_with_eof = tokenizer.encode_with_endoftext(text)
        
        # Should have one more token
        assert len(tokens_with_eof) == len(tokens_normal) + 1
        
        # The first part should match
        assert tokens_with_eof[:-1] == tokens_normal
        
        # The last token should be endoftext
        assert tokens_with_eof[-1] == tokenizer._endoftext_id
    
    def test_encode_with_endoftext_empty_string(self, tokenizer):
        """Encoding empty string with endoftext should just be the endoftext token."""
        tokens = tokenizer.encode_with_endoftext("")
        assert len(tokens) == 1
        assert tokens[0] == tokenizer._endoftext_id


class TestTokenizerEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_very_long_text(self, tokenizer):
        """Should handle very long text without crashing."""
        long_text = "word " * 10000
        tokens = tokenizer.encode(long_text)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert decoded == long_text
    
    def test_unicode_text(self, tokenizer):
        """Should handle various unicode characters."""
        unicode_text = "Hello 世界 мир العالم"
        tokens = tokenizer.encode(unicode_text)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert decoded == unicode_text
    
    def test_newlines_and_tabs(self, tokenizer):
        """Should handle whitespace characters."""
        text = "line1\nline2\ttabbed"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_multiple_spaces(self, tokenizer):
        """Should handle multiple consecutive spaces."""
        text = "word1     word2"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_punctuation_only(self, tokenizer):
        """Should encode pure punctuation."""
        punct = "!?.,;:'\"-()[]{}@#$%^&*+=<>/"
        tokens = tokenizer.encode(punct)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert decoded == punct


class TestTokenizerConsistency:
    """Tests for consistency and determinism."""
    
    def test_encode_decode_multiple_roundtrips(self, tokenizer):
        """Multiple encode/decode cycles should be consistent."""
        original = "The quick brown fox jumps over the lazy dog"
        text = original
        for _ in range(5):
            tokens = tokenizer.encode(text)
            text = tokenizer.decode(tokens)
        assert text == original
    
    def test_token_count_consistency(self, tokenizer):
        """Token count should be consistent for the same text."""
        text = "consistency test"
        counts = [len(tokenizer.encode(text)) for _ in range(10)]
        assert len(set(counts)) == 1  # All counts should be identical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
