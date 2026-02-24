
import torch
import torch.nn as nn
from configs.device_config import DeviceConfig


class DeviceManager:
    def __init__(self, config):
        self.config = config
        self.device = config.get_device()

    def to_device(self, tensor):
        """
        将张量或容器移动到指定设备

        Args:
            tensor: 输入张量、字典或列表

        Returns:
            移动到设备后的对象
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, dict):
            return {k: self.to_device(v) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return [self.to_device(item) for item in tensor]
        elif isinstance(tensor, tuple):
            return tuple(self.to_device(item) for item in tensor)
        return tensor

    def data_parallel(self, model):
        """
        对模型应用数据并行

        Args:
            model: PyTorch 模型

        Returns:
            应用 DataParallel 后的模型
        """
        if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            model = nn.DataParallel(model)
        return model

    @staticmethod
    def check_cuda_available():
        """
        检查 CUDA 是否可用

        Returns:
            bool: CUDA 是否可用
        """
        return torch.cuda.is_available()

    @staticmethod
    def get_gpu_info():
        """
        获取 GPU 信息

        Returns:
            Dict[str, Union[int, str, float]]: GPU 信息字典
                - 'device_count': GPU 数量
                - 'device_names': GPU 名称列表
                - 'memory_available': 可用显存
        """
        if not torch.cuda.is_available():
            return {
                'device_count': 0,
                'device_names': [],
                'memory_available': 0.0
            }
        
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        
        memory_available = 0.0
        if device_count > 0:
            torch.cuda.synchronize()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_available = (memory_total - memory_allocated) / (1024 ** 3)
        
        return {
            'device_count': device_count,
            'device_names': device_names,
            'memory_available': memory_available
        }
