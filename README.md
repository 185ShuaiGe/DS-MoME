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
  - **100% 冻结** LLM 主干网络，彻底移除 LoRA 等微调机制，并引入 4-bit 量化（NF4）极大降低显存消耗。
  - 全网仅允许**底层伪影流（浅层 CNN）**与**跨模态融合机制（MoME）**进行梯度更新。
- **物理伪影与语义双流 (Dual-Stream Encoder)**：结合深层语义推理（CLIP）与底层物理伪影提取（固定 SRM 滤波器 + 浅层 CNN），从物理根源上切断“语义泄漏”。
- **动态模态专家混合融合 (MoME Fusion)**：视觉驱动的动态专家混合融合机制，根据图像自身特性自动路由调用不同的融合专家。
- **丰富的实验与测试管线 (Rich Pipelines)**：内置早停机制（Early Stopping）、支持动态在内存中注入扰动（JPEG/高斯模糊）的鲁棒性测试框架，并能自动生成 ROC、PR 及 Loss 训练曲线图表。

---

## 📁 目录结构 (Directory Structure)

```text
DS-MoME-dev/
├── main.py                     # 项目主入口 (训练/验证/推理)，统一参数调度
├── run_linear_probe.py         # 纯 CLIP 线性探测基准测试脚本
├── test_ds_fdmas.py            # FDMAS 标准化跨库独立测试脚本
├── configs/                    # 配置文件目录 (模型/设备/路径/消融参数配置)
├── data/                       # 数据集加载逻辑与处理
│   └── dataset_loader.py       # 包含 9:1 动态划分与数据增强逻辑
├── models/                     # 核心模型架构与训练逻辑
│   ├── ds_mome.py              # DS-MoME 整体网络组装
│   ├── dual_stream_encoder.py  # 双流编码器 (CLIP + SRM-CNN)
│   ├── mome_fusion.py          # 动态混合模态专家融合网络
│   ├── llm_infer.py            # LLM 推理接口 (支持 4-bit 量化装载)
│   ├── linear_probe_clip.py    # 仅含线性分类头的 CLIP 基线模型
│   ├── trainer.py              # 单阶段训练器 (支持早停机制)
│   └── validator.py            # 模型验证器
├── scripts/                    # 自动化与测试脚本
│   ├── run_pipeline.sh         # 自动化训练 + 寻优 + FDMAS 测试流水线
│   ├── ablation.sh             # 全套消融实验自动化挂机脚本
│   ├── test_ablation_only.sh   # 独立测试已有消融权重的脚本
│   └── robust_ds_fdmas_dynamic.py # 动态鲁棒性测试 (内存注入扰动，0硬盘占用)
├── utils/                      # 工具类 (显卡监控、Token 处理、日志记录)
│   ├── log_utils.py            # 统一的双向日志系统 (控制台+文件)
│   └── metrics_utils.py        # 评测指标计算及自动化图表绘制 (ROC/PR等)
├── gpu_watcher.sh              # GPU 显存监控与排队挂机脚本
├── download_*.py               # LLM、CLIP 与数据集的国内镜像加速下载脚本
├── environment.yml             # Conda 环境配置
└── requirements.txt            # Python 依赖清单
```

---

## ⚙️ 环境配置 (Installation)

确保你的环境中安装了 PyTorch 及 Hugging Face 相关基础库，**强烈建议**使用提供的 conda 配置文件进行安装：

```bash
# 1. 使用 conda 创建环境 (推荐)
conda env create -f environment.yml
conda activate c2ppy39

# 2. 或者使用 pip 安装核心依赖
pip install -r requirements.txt
```

*注意：本项目深度依赖 `bitsandbytes` 进行 LLM 的 4-bit 量化以防显存溢出 (OOM)，请确保其正确安装并在支持 CUDA 的 Linux 环境下运行。*

---

## 🚀 快速开始 (Quick Start)

我们提供了高度集成的 `main.py` 作为入口，并通过自动化脚本大幅简化了训练与测试流程。

### 1. 自动化流水线 (强烈推荐)

使用 `run_pipeline.sh` 可以一键完成 **“启动训练 -> 自动寻找最佳权重 -> 在 FDMAS 数据集上测试”** 的完整闭环。

```bash
# 赋予执行权限（首次）
chmod +x scripts/run_pipeline.sh

# 运行流水线 (参数：指定的 GPU_ID)
./scripts/run_pipeline.sh 0
```

### 2. 手动启动训练 (Training)

默认使用 BCE Loss 进行单阶段二分类训练，并内置了早停机制（`--patience`）。
```bash
python main.py \
    --mode train \
    --gpu_id 0 \
    --batch_size 4 \
    --num_epochs 10 \
    --lr 1e-4 \
    --patience 3 \
    --ablation final
```
*训练结束后，程序会在 `outputs/` 目录下自动生成对应的 Loss 和 AUC 走势折线图。*

### 3. FDMAS 跨库评估 (Testing)

用于在测试集上独立评估已保存的模型权重，自动输出 `ACC`, `RACC`, `FACC`, `AP` 指标：
```bash
python test_ds_fdmas.py --gpu_id 0 --checkpoint ./weights/final/your_best_model.pt
```

---

## 🔬 实验与深度测试 (Advanced Evaluation)

除了基础测试外，本项目还提供了多种深度验证机制：

### 1. 动态鲁棒性测试 (Dynamic Robustness Test)
直接在内存中施加 JPEG 压缩和高斯模糊扰动（不修改原始硬盘文件），验证模型在图像降质情况下的表现。
```bash
python scripts/robust_ds_fdmas_dynamic.py --gpu_id 0
```

### 2. 纯 CLIP 线性探测 (Linear Probe Baseline)
用于确立纯语义特征（CLIP ViT-L/14 + Linear Layer）在当前检测任务上的基线水平。
```bash
python run_linear_probe.py --gpu_id 0 --batch_size 32
```

### 3. 自动化消融实验 (Ablation Studies)
在 `configs/ablation_config.py` 中定义了以下组别：
* **A**: 仅保留高级语义流（CLIP）。
* **B**: 仅保留底层伪影流（SRM + 浅层 CNN）。
* **C1/C2/C3**: 仅激活特定通道的 SRM 滤波器。
* **D**: 退化为传统 MLP 拼接（禁用 MoME）。
* **final**: 完整架构。

支持使用自动化脚本一键跑通所有消融实验及测试：
```bash
chmod +x scripts/ablation.sh
./scripts/ablation.sh
```

---

## 📊 评估指标与可视化 (Metrics & Visualization)

本框架专注于高精度的真伪鉴别，`utils/metrics_utils.py` 会在验证结束后自动计算并**绘制保存图表**至 `outputs/` 目录：
* **ACC (Accuracy)**: 整体鉴别准确率。
* **AP (Average Precision)**: 平均精度，衡量模型在不同阈值下的综合排序能力。
* **RACC (Real Accuracy)**: 真实图像的召回率。
* **FACC (Fake Accuracy)**: AI 生成图像的召回率。
* **ROC & PR 曲线图**: 训练完成后自动生成。

---

## 📝 引用 (Citation)

如果您在研究中使用了本代码或参考了 DS-MoME 架构，请引用我们的研究工作：
*(待更新)*
```