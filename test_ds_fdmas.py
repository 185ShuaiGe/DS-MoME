import os
import argparse
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 0. 解析参数并隔离显卡
# =========================================================
parser = argparse.ArgumentParser(description="Test DSMoME on FDMAS dataset")
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号')
parser.add_argument('--checkpoint', type=str, required=True, help='模型权重(.pt文件)的路径')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 🎯 核心修复：在 import 模型模块之前，提前配置好消融开关！
# =========================================================
# 这样底层网络在 import 时，就能按照真实的维度 (如 B组的 512) 进行初始化
from configs.ablation_config import AblationConfig

MODEL_PATH = args.checkpoint
ckpt_name = os.path.basename(MODEL_PATH)
match = re.match(r'^([A-Za-z0-9]+)[-_]', ckpt_name)
exp_id = match.group(1) if match else "final"

print(f"🧩 [预初始化] 自动识别到消融组别: [{exp_id}]，正在配置全局网络拓扑...")
AblationConfig.EXPERIMENT_ID = exp_id
AblationConfig.apply_config()

# =========================================================
# 1. 此时再导入深度学习库和你的业务代码，它们读取到的就是正确配置了
# =========================================================
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME

TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test/'
BATCH_SIZE = 4
TEXT_PROMPT = "<image>\nAnalyze this image and determine if it is real or AI-generated. Please provide your reasoning."

def main():
    print(f"🚀 开始测试 FDMAS 数据集...")
    print(f"🖥️  使用物理 GPU 编号: {args.gpu_id}")
    print(f"📂 数据集路径: {TEST_ROOT}")
    print(f"⚖️  模型路径: {MODEL_PATH}")
    
    model_config = ModelConfig()
    device_config = DeviceConfig()
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)
    path_config = PathConfig()
    device = device_config.get_device()

    print("⏳ 正在初始化 DSMoME 模型...")
    # 现在的模型实例，绝对拥有和该组别完全匹配的特征维度！
    model = DSMoME(model_config, device_config, path_config)

    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    
    print("⏳ 正在加载权重...")
    if os.path.exists(MODEL_PATH):
        # 如果维度依然不对，这里会直接崩溃退出，不再“带伤上阵”
        model.load_checkpoint(MODEL_PATH)
    else:
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model.eval()
    print("✅ 模型加载完成，维度匹配成功！")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    sub_datasets = sorted([d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))])
    
    print("\n" + "="*70)
    print(f"{'Dataset':<25} | {'ACC (%)':<10} | {'RACC (%)':<10} | {'FACC (%)':<10} | {'AP (%)':<10}")
    print("-" * 70)

    all_acc, all_ap, all_racc, all_facc = [], [], [], []

    for dataset_name in sub_datasets:
        dataset_path = os.path.join(TEST_ROOT, dataset_name)
        
        try:
            dataset = ImageFolder(root=dataset_path, transform=transform)
        except Exception as e:
            print(f"{dataset_name:<25} | Error      | Error      | Error      | Error: {e}")
            continue

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                prompts = [TEXT_PROMPT] * images.size(0)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images, prompts, text_guidance=prompts)
                    logits = outputs.get('detection_logits', None)
                    
                    if logits is not None:
                        probs = torch.sigmoid(logits.float()).squeeze()
                        if probs.ndim == 0: probs = probs.unsqueeze(0)
                        y_pred.extend(probs.cpu().tolist())
                        y_true.extend(labels.tolist())

        if len(y_true) == 0: continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred).reshape(-1, 1)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        real_mask = y_true == 0
        real_total = np.sum(real_mask)
        racc = (np.sum((y_pred[real_mask] <= 0.5).astype(int)) / real_total) if real_total > 0 else 0.0
        
        fake_mask = y_true == 1
        fake_total = np.sum(fake_mask)
        facc = (np.sum((y_pred[fake_mask] > 0.5).astype(int)) / fake_total) if fake_total > 0 else 0.0

        all_acc.append(acc)
        all_ap.append(ap)
        all_racc.append(racc)
        all_facc.append(facc)
        
        print(f"{dataset_name:<25} | {acc*100:5.2f}      | {racc*100:5.2f}      | {facc*100:5.2f}      | {ap*100:5.2f}")

    print("-" * 70)
    if all_acc:
        print(f"{'MEAN':<25} | {np.mean(all_acc)*100:5.2f}      | {np.mean(all_racc)*100:5.2f}      | {np.mean(all_facc)*100:5.2f}      | {np.mean(all_ap)*100:5.2f}")
    print("=" * 70)

if __name__ == '__main__':
    main()