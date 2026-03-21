# DS-MoME: A Dual-Stream Mixture of Modality Experts for Image Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

DS-MoME (Dual-Stream Mixture of Modality Experts) 是一个专为**人工智能生成图像（AIGI）检测**设计的极简、高效且极具泛化能力的架构。

本研究深刻剖析了当前多模态大语言模型（MLLMs）在伪造检测领域的应用局限，创造性地提出**“放弃生成式解释、回归二元分类本质”**的理念，彻底免除强迫大模型输出自然语言所带来的“对齐税（Alignment Tax）”。

## 🌟 核心特性 (Key Features)

- **单阶段纯二分类训练 (Single-Stage Binary Classification)**：摒弃繁琐的多阶段指令微调（SFT/DPO）与文本生成任务，仅使用标准的二元交叉熵损失函数（BCE Loss）直接优化大模型的隐藏状态输出。
- **极致的参数冻结策略 (Extreme Freezing Strategy)**：
  - **100% 冻结** CLIP 语义流参数。
  - **100% 冻结** LLM 主干网络（彻底移除 LoRA 等微调机制），并引入 4-bit 量化极大降低显存消耗。
  - 全网仅允许**底层伪影流（浅层 CNN）**与**跨模态融合机制（MoME）**进行梯度更新（约 15M 可训练参数）。
- **物理伪影与语义双流 (Dual-Stream Encoder)**：结合深层语义推理（CLIP）与底层物理伪影提取（固定 SRM 滤波器 + 浅层 CNN），从物理根源上切断“语义泄漏”。
- **动态模态专家混合融合 (MoME Fusion)**：视觉驱动的动态专家混合融合机制，根据图像自身特性自动路由调用不同的融合专家，完美契合处理多维度取证线索的需求。

---

## 📁 目录结构 (Directory Structure)

```text
DS-MoME/
├── main.py                     # 项目主入口 (训练/验证/推理)
├── test_ds_fdmas.py            # FDMAS 等跨库/鲁棒性独立测试脚本
├── configs/                    # 配置文件目录
│   ├── model_config.py         # 模型超参数、消融实验开关配置
│   ├── device_config.py        # 设备与并行配置
│   └── path_config.py          # 数据集与权重路径配置
├── models/                     # 核心模型架构
│   ├── ds_mome.py              # DS-MoME 整体网络组装
│   ├── dual_stream_encoder.py  # 双流编码器 (CLIP + SRM-CNN)
│   ├── mome_fusion.py          # 动态混合模态专家融合网络
│   ├── llm_infer.py            # LLM 推理接口 (支持 4-bit 量化)
│   ├── trainer.py              # 单阶段二分类训练器
│   └── validator.py            # 模型验证器
├── utils/                      # 工具类
│   ├── log_utils.py            # 日志记录与显存追踪
│   └── metrics_utils.py        # 评测指标计算 (ACC, AP, RACC, FACC)
└── data/                       # 数据处理
    └── dataset_loader.py       # AIGI 数据集加载器

```

---

## ⚙️ 环境依赖 (Installation)

确保你的环境中安装了 PyTorch 及 Hugging Face 相关基础库，并额外安装量化支持库以防 LLM 导致显存溢出 (OOM)：

```bash
pip install torch torchvision
pip install transformers accelerate
# 必须安装的 4-bit 量化支持库，大幅降低大模型显存占用
pip install bitsandbytes 
pip install scikit-learn tqdm pillow

```

---

## 🚀 快速开始 (Quick Start)

我们提供了统一的命令行入口 `main.py`，支持通过 `--gpu_id` 参数物理隔离显卡，避免多卡环境下的设备冲突。

### 1. 模型训练 (Training)

默认使用 BCE Loss 进行单阶段训练。训练结束后会自动在终端打印详细的**显存占用追踪报告**。

```bash
python main.py --mode train --gpu_id 0 --batch_size 4 --num_epochs 3 --lr 5e-5 --ablation final

```

### 2. 单张图像推理 (Single Image Inference)

```bash
python main.py --mode inference --gpu_id 0 --image_path /path/to/your/image.jpg --checkpoint /path/to/your/checkpoint_best.pt
```

### 3. 批量图像推理 (Batch Inference)

结果将自动保存至 `outputs/` 目录下的 JSON 文件中。

```bash
python main.py --mode inference --gpu_id 1 --image_dir /path/to/your/image_folder/ --checkpoint /path/to/your/checkpoint_best.pt

```

### 4. 标准化数据集测试 (Testing on Datasets)

对于如 FDMAS 等包含真实(Real)与伪造(Fake)子目录的标准数据集，可使用专用的测试脚本，它将自动输出 `ACC`, `RACC`, `FACC`, `AP` 等指标：

```bash
python test_ds_fdmas.py --gpu_id 0

```

---

## 🔬 消融实验 (Ablation Studies)

为了严谨验证 DS-MoME 架构中双流特征提取与动态融合机制的有效性，本项目内置了高度模块化的消融实验框架。所有的网络拓扑结构切换已统一收敛至配置中枢，无需手动修改底层模型代码。

### 1. 实验组别定义
在 `configs/ablation_config.py` 中，我们定义了以下标准的消融实验组别：

* **基线 A (Baseline A)**：仅保留高级语义流（CLIP），弃用伪影流与融合模块。用于验证面对复杂合成图像时，单一语义特征的局限性。
* **基线 B (Baseline B)**：仅保留底层伪影流（SRM + 浅层 CNN），弃用语义流与融合模块。用于验证缺乏全局视觉语境时的检测盲区。
* **消融 C系列 (Ablation C1/C2/C3)**：频域特征提取鲁棒性测试。通过动态掩码（Masking）分别仅激活第 1、2、3 个 SRM 滤波器。用于证明多通道高频特征提取在防止特征同质化上的不可替代性。
* **消融 D (Ablation D)**：多模态融合机制退化测试。弃用动态的模态专家混合（MoME）网络，退化为使用传统的双层 MLP 进行特征硬拼接。用于验证 MoME 在克服“模态干扰”与“注意力稀释”方面的优越性。
* **最终形态 (final)**：完整的 DS-MoME 架构（双流并发 + 3 通道 SRM + 动态 MoME 融合）。

### 2. 一键自动化运行
我们提供了高度自动化的 Shell 脚本，支持在无人值守的情况下依次完成各组别的训练、最优权重保存与独立评估。

**自动化流水线 (训练 + 评估)：**
支持一键跑通 A, B, C1, C2, C3, D, final 的全套流程，自动打标时间戳并按组别隔离保存权重。
```bash
# 赋予执行权限（首次运行需要）
chmod +x ./scripts/ablation.sh

# 启动全流程消融实验
./scripts/ablation.sh



---

## 📊 评估指标说明 (Metrics)

本框架专注于高精度的真伪鉴别，主要评估指标包括：

* **ACC (Accuracy)**: 整体检测准确率。
* **AP (Average Precision)**: 平均精度，衡量模型在不同阈值下的综合表现。
* **RACC (Real Accuracy)**: 真实图像（标签 0）的鉴别准确率。
* **FACC (Fake Accuracy)**: AI 生成图像（标签 1）的鉴别准确率。

---

## 📝 引用 (Citation)

如果您在研究中使用了本代码或参考了 DS-MoME 架构，请引用我们的研究报告：
*(待更新)*

```

```