
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig


class LLMInference(nn.Module):
    def __init__(self, config: ModelConfig, device_config: DeviceConfig):
        super().__init__()
        self.config = config
        self.device_config = device_config
        self.llm_model = None
        self.tokenizer = None
        self.lora_config = None
        
    def forward(self, input_ids, attention_mask, vision_tokens=None):
        pass
    
    def generate(self, prompt, vision_tokens=None, max_new_tokens=256):
        pass
    
    def _init_lora(self):
        pass
    
    def _apply_lora(self):
        pass
