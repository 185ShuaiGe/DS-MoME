
from transformers import AutoTokenizer
from configs.model_config import ModelConfig


class TokenUtils:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        
    def encode_text(self, text, max_length=None, return_tensors="pt"):
        pass
    
    def decode_tokens(self, token_ids, skip_special_tokens=True):
        pass
    
    def pad_sequences(self, sequences, padding_side="right"):
        pass
    
    def _init_tokenizer(self):
        pass
