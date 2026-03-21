import os
import argparse

# =========================================================
# 0. 解析参数并隔离显卡 (必须在 import torch 之前执行)
# =========================================================
parser = argparse.ArgumentParser(description="Train & Test Linear Probe CLIP")
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的 GPU 编号 (例如 0 或 1)')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小 (Linear Probe 显存占用小，可适当调大)')
args = parser.parse_args()

# 强制系统只对当前 Python 进程暴露指定的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================
# 导入深度学习库
# =========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from configs.model_config import ModelConfig
from configs.device_config import DeviceConfig
from configs.path_config import PathConfig
from data.dataset_loader import get_holmes_dataloaders
from models.linear_probe_clip import LinearProbeCLIP

# =========================================================
# 核心验证函数：复用 test_ds_fdmas.py 的逻辑
# =========================================================
def evaluate_fdmas(model, device, test_root, batch_size=8):
    print("\n" + "="*70)
    print(f"🚀 开始在 FDMAS 数据集上进行测试评估...")
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    sub_datasets = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
    
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

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                
                # 保持测试时使用混合精度
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)  # Linear Probe 不需要 text_prompt
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

    print("-" * 70)
    if all_acc:
        mean_acc = np.mean(all_acc) * 100
        mean_ap = np.mean(all_ap) * 100
        mean_racc = np.mean(all_racc) * 100  
        mean_facc = np.mean(all_facc) * 100  
        print(f"{'MEAN OVERALL':<25} | {mean_acc:5.2f}      | {mean_racc:5.2f}      | {mean_facc:5.2f}      | {mean_ap:5.2f}")
    print("=" * 70 + "\n")


# =========================================================
# 统一流程：训练 + FDMAS 测试
# =========================================================
def main():
    config = ModelConfig()
    path_config = PathConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 强制覆盖测试集路径为 FDMAS
    TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test/'

    print(f"=== 纯 CLIP 线性探测 (Linear Probe) 基准实验 ===")
    print(f"使用设备: {device}")
    
    # 1. 初始化模型并移至设备
    model = LinearProbeCLIP(config).to(device)

    # 2. 严格遵循控制要求：设定固定参数
    epochs = 5
    learning_rate = 5e-5
    
    # 关键：优化器中仅传入分类器（nn.Linear）的参数
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 3. 加载训练数据 (丢弃验证集，因为我们要在 FDMAS 上测)
    print("⏳ 正在加载 Holmes 训练数据...")
    train_loader, _ = get_holmes_dataloaders(path_config, config, batch_size=args.batch_size)

    print(f"\n🚀 开始训练 | Epochs: {epochs} | LR: {learning_rate}")

    # 4. 训练与测试一体化循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        # 兼容 AIGIDataset 返回 4 个元素
        for batch_idx, (images, labels, _, _) in enumerate(train_bar):
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            
            # 开启混合精度加快纯分类头的训练
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                logits = outputs['detection_logits'].squeeze()
                
                # 处理单张图片 batch 的边界情况
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                    
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} 训练完成 | Avg Train Loss: {avg_train_loss:.4f}")
        
    # 5. 在训练5个 Epoch 后，直接调用 FDMAS 的严苛测试流程
    evaluate_fdmas(model, device, test_root=TEST_ROOT)

if __name__ == "__main__":
    main()