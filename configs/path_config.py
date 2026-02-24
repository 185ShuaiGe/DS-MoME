
import os

class PathConfig:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    weights_dir = os.path.join(project_root, "weights")
    outputs_dir = os.path.join(project_root, "outputs")
    logs_dir = os.path.join(project_root, "logs")
    pretrained_clip_path = None
    pretrained_resnet_path = None
    pretrained_llm_path = None
    checkpoint_path = os.path.join(weights_dir, "plaa_mllm_checkpoint.pt")
