
import torch
from transformers import AutoTokenizer
from configs.model_config import ModelConfig


class TokenUtils:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self._init_tokenizer()

    def encode_text(
        self,
        text,
        max_length=None,
        return_tensors="pt"
    ):
        """
        将文本编码为 token ID

        Args:
            text: 输入文本字符串或字符串列表
            max_length: 最大序列长度
            return_tensors: 返回张量类型，'pt' 为 PyTorch，'np' 为 NumPy，'tf' 为 TensorFlow

        Returns:
            Dict[str, Union[list, torch.Tensor]]: 编码结果
                - 'input_ids': token ID 列表或张量
                - 'attention_mask': 注意力掩码列表或张量
        """
        if max_length is None:
            max_length = self.config.max_seq_len
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )

    def decode_tokens(
        self,
        token_ids,
        skip_special_tokens=True
    ):
        """
        将 token ID 解码为文本

        Args:
            token_ids: token ID 列表或张量
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            str: 解码后的文本
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids[0]
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def pad_sequences(
        self,
        sequences,
        padding_side="right"
    ):
        """
        对序列进行填充

        Args:
            sequences: 序列列表
            padding_side: 填充方向，'left' 或 'right'

        Returns:
            Dict[str, torch.Tensor]: 填充后的结果
                - 'input_ids': 填充后的 input_ids
                - 'attention_mask': 注意力掩码
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        max_len = max(len(seq) for seq in sequences)
        
        input_ids = []
        attention_masks = []
        
        for seq in sequences:
            pad_len = max_len - len(seq)
            
            if padding_side == "right":
                padded_seq = seq + [self.tokenizer.pad_token_id] * pad_len
                mask = [1] * len(seq) + [0] * pad_len
            else:
                padded_seq = [self.tokenizer.pad_token_id] * pad_len + seq
                mask = [0] * pad_len + [1] * len(seq)
            
            input_ids.append(padded_seq)
            attention_masks.append(mask)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks)
        }

    def _init_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            pass
