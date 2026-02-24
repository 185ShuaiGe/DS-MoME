
import torch
from configs.device_config import DeviceConfig


class DeviceManager:
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.device = config.get_device()
        
    def to_device(self, tensor):
        pass
    
    def data_parallel(self, model):
        pass
    
    @staticmethod
    def check_cuda_available():
        pass
    
    @staticmethod
    def get_gpu_info():
        pass
