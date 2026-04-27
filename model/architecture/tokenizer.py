from typing import List
from tokenizers import Tokenizer
from pathlib import Path


class CustomTokenizer:
    
    ENDOFTEXT_TOKEN = "[END]"
    
    def __init__(self):
        tokenizer_path = Path(__file__).parent.parent / "architecture" / "tokenizer" / "mini-gpt-tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._vocab_size = self.tokenizer.get_vocab_size()
        self._endoftext_id = self.tokenizer.token_to_id(self.ENDOFTEXT_TOKEN)
    
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        return self.tokenizer.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode_with_endoftext(self, text: str) -> List[int]:
        tokens = self.encode(text)
        return tokens + [self._endoftext_id]