
import argparse
import torch
from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.plaa_mllm import PLAAMLLM
from utils.log_utils import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="PLAA-MLLM AI Generated Image Detection")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "val", "inference"])
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image for inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    model_config = ModelConfig()
    device_config = DeviceConfig()
    path_config = PathConfig()
    
    logger = Logger(name="PLAA_MLLM_Main", log_dir=path_config.logs_dir)
    
    model = PLAAMLLM(model_config, device_config, path_config)
    
    device = device_config.get_device()
    model = model.to(device)
    
    if args.mode == "train":
        train(model, args, logger)
    elif args.mode == "val":
        validate(model, args, logger)
    elif args.mode == "inference":
        inference(model, args, logger)


def train(model, args, logger):
    pass


def validate(model, args, logger):
    pass


def inference(model, args, logger):
    pass


if __name__ == "__main__":
    main()
