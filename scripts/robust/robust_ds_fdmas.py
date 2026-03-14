import os
import sys
import argparse
import logging
import datetime

# =========================================================
# 0. 路径挂载与系统配置
# =========================================================
# 动态将项目根目录加入系统路径，确保能识别 configs 和 models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Robustness Test DSMoME on Corrupted FDMAS datasets")
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号 (例如 0 或 1)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 1. 导入依赖与本地包
# =========================================================
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import warnings

from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from models.ds_mome import DSMoME

warnings.filterwarnings('ignore')

# =========================================================
# 2. 双端日志重定向设置 (控制台 + Log文件)
# =========================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # 确保 log 文件夹存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 设置日志路径
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'robust_test_fdmas.log')
sys.stdout = Logger(LOG_FILE)

# =========================================================
# 3. 基础参数配置
# =========================================================
BASE_DATASET_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas'
MODEL_PATH = '/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/weights/checkpoint_best.pt'
BATCH_SIZE = 4
TEXT_PROMPT = "<image>\nAnalyze this image and determine if it is real or AI-generated. Please provide your reasoning."

# 需要测试的 6 个降质数据集目录名
ROBUST_TEST_DIRS = [
    'test_jpeg_90',
    'test_jpeg_75',
    'test_jpeg_50',
    'test_blur_1.0',
    'test_blur_2.0',
    'test_blur_3.0'
]

# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("\n" + "#"*70)
    print(f"🚀 [START] 鲁棒性批量测试 - 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  使用物理 GPU 编号: {args.gpu_id}")
    print(f"⚖️  模型路径: {MODEL_PATH}")
    print(f"📄 日志保存路径: {LOG_FILE}")
    print("#"*70 + "\n")
    
    model_config = ModelConfig()
    device_config = DeviceConfig()
    
    device_config.gpu_ids = [0]
    device_config.cuda_visible_devices = str(args.gpu_id)
    path_config = PathConfig()
    device = device_config.get_device()

    print("⏳ 正在初始化 DSMoME 模型...")
    model = DSMoME(model_config, device_config, path_config)

    model.dual_stream_encoder = model.dual_stream_encoder.to(device)
    model.mome_fusion = model.mome_fusion.to(device)
    model.vision_token_proj = model.vision_token_proj.to(device)
    model.detection_head = model.detection_head.to(device)
    
    print("⏳ 正在加载权重...")
    if os.path.exists(MODEL_PATH):
        model.load_checkpoint(MODEL_PATH)
    else:
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model.eval()
    print("✅ 模型加载完成！\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # ---------------------------------------------------------
    # 核心测试循环：遍历 6 个扰动数据集
    # ---------------------------------------------------------
    for dataset_folder in ROBUST_TEST_DIRS:
        test_root = os.path.join(BASE_DATASET_ROOT, dataset_folder)
        print("\n" + "★"*70)
        print(f"📂 正在测试扰动数据集: 【 {dataset_folder.upper()} 】")
        print(f"   路径: {test_root}")
        print("★"*70)

        if not os.path.exists(test_root):
            print(f"❌ 目录不存在: {test_root}，请先运行 generate_robust_testset.py")
            continue

        sub_datasets = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
        
        print("\n" + "="*70)
        print(f"{'Dataset':<25} | {'ACC (%)':<10} | {'RACC (%)':<10} | {'FACC (%)':<10} | {'AP (%)':<10}")
        print("-" * 70)

        all_acc, all_ap, all_racc, all_facc = [], [], [], []

        for dataset_name in sub_datasets:
            dataset_path = os.path.join(test_root, dataset_name)
            
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
                            if probs.ndim == 0:
                                probs = probs.unsqueeze(0)
                                
                            y_pred.extend(probs.cpu().tolist())
                            y_true.extend(labels.tolist())

            if len(y_true) == 0:
                continue

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

        # 打印当前扰动数据集的平均值
        print("-" * 70)
        if all_acc:
            mean_acc, mean_ap = np.mean(all_acc) * 100, np.mean(all_ap) * 100
            mean_racc, mean_facc = np.mean(all_racc) * 100, np.mean(all_facc) * 100
            print(f"{'MEAN':<25} | {mean_acc:5.2f}      | {mean_racc:5.2f}      | {mean_facc:5.2f}      | {mean_ap:5.2f}")
        print("=" * 70)

    print("\n🏁 所有鲁棒性测试执行完毕！")

if __name__ == '__main__':
    main()