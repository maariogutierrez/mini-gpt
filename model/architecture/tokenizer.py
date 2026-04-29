from typing import List
import tiktoken


class CustomTokenizer:
    
    ENDOFTEXT_TOKEN = "[END]"
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self.encoding.n_vocab
        self._endoftext_id = self.encoding.encode(
            self.ENDOFTEXT_TOKEN, allowed_special={self.ENDOFTEXT_TOKEN}
        )[0]
    
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        return self.encoding.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode_with_endoftext(self, text: str) -> List[int]:
        tokens = self.encode(text)
        return tokens + [self._endoftext_id]