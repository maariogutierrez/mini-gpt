import tiktoken
from typing import List


class Tokenizer:
    """Clean wrapper around tiktoken for encoding/decoding text."""
    
    ENDOFTEXT_TOKEN = "<|endoftext|>"
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize tokenizer with specified encoding.
        
        Args:
            encoding_name: Name of the tiktoken encoding (default: cl100k_base used by GPT models)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self.encoding.n_vocab
        self._endoftext_id = self.encoding.encode(
            self.ENDOFTEXT_TOKEN, allowed_special={self.ENDOFTEXT_TOKEN}
        )[0]
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded string
        """
        if not tokens:
            return ""
        return self.encoding.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def encode_with_endoftext(self, text: str) -> List[int]:
        """
        Encode text and append endoftext token (useful for document separation).
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs with endoftext token appended
        """
        tokens = self.encode(text)
        return tokens + [self._endoftext_id]

