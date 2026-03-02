
import torch

class DeviceConfig:
    use_gpu = True
    gpu_ids = [0, 1]                # 修改：增加显卡 1
    cuda_visible_devices = "0, 1"    # 修改：增加显卡 1
    dtype = torch.float32
    use_amp = True
    
    @classmethod
    def get_device(cls):
        if cls.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{cls.gpu_ids[0]}")       #保持返回 cuda:0 作为主设备即可
        else:
            return torch.device("cpu")
