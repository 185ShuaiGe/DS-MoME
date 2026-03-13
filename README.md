
# PLAA-MLLM_AIGI_Detection

PLAA-MLLM (Prompt-Learned, Artifact-Augmented Multimodal Large Language Model) 是一个用于 AI 生成图像检测的项目。

## 目录结构

```
PLAA-MLLM_AIGI_Detection/
├── main.py               # 项目入口
├── configs/              # 配置文件目录
│   ├── model_config.py   # 模型超参数配置
│   ├── device_config.py  # GPU/CPU 设备配置
│   └── path_config.py    # 路径配置
├── models/               # 核心模型架构目录
│   ├── dual_stream_encoder.py    # 双流多尺度视觉特征提取模块
│   ├── mome_fusion.py            # 视觉驱动的动态专家混合融合网络
│   ├── llm_infer.py       # LLM 推理核心
│   ├── plaa_mllm.py       # PLAA-MLLM 整体组装
│   ├── trainer.py         # 两阶段训练器
│   └── validator.py       # 模型验证器
├── utils/                # 工具类目录
│   ├── device_utils.py   # 设备管理工具
│   ├── log_utils.py      # 日志工具
│   ├── token_utils.py    # 分词工具
│   └── metrics_utils.py  # 多维度指标计算器
├── data/                 # 数据集目录
│   ├── dataset_loader.py # 数据集加载器
│   ├── holmes_dataset/   # Holmes 数据集
│   ├── train/            # 训练集
│   ├── val/              # 验证集
│   └── test/             # 测试集
├── weights/              # 模型权重目录
├── outputs/              # 输出结果目录
│   └── metrics/          # 指标可视化结果
├── logs/                 # 日志目录
├── requirements.txt      # 环境依赖清单
└── README.md             # 项目说明文档
```

## 文件调用关系

1. **main.py** - 项目入口，解析命令行参数并初始化各模块
2. **configs/** - 所有配置文件被其他模块导入使用
3. **models/plaa_mllm.py** - 核心组装类，整合三个主要模块：
   - `dual_stream_encoder.py` - 双流编码器
   - `mome_fusion.py` - 视觉驱动的动态专家混合融合网络
   - `llm_infer.py` - LLM 推理
4. **utils/** - 工具类被各模块按需调用

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n plaa_mllm python=3.9
conda activate plaa_mllm
```

### 2. 安装 PyTorch 2.6 (CUDA 11.8)

```bash
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

或 CUDA 12.1 版本：

```bash
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 两阶段训练

#### 单阶段训练

**阶段 1：特征对齐与路由预训练**

```bash
python main.py --mode train --train_stage 1 --batch_size 8 --num_epochs 10 --lr 1e-4
```

**阶段 2：基于专家解释的深度指令微调**

```bash
python main.py --mode train --train_stage 2 --batch_size 1 --num_epochs 5 --lr 5e-5 --checkpoint weights/checkpoint_stage1_best.pt
```

### 模型验证

```bash
python main.py --mode val --checkpoint weights/checkpoint_stage2_best.pt --batch_size 8
```

验证结果将保存到：
- `outputs/validation_results.json` - 详细验证结果
- `outputs/metrics/` - 指标可视化图表（ROC 曲线、PR 曲线、指标柱状图）

### 单张图像推理

```bash
python main.py --mode inference --image_path /path/to/test_image.jpg --checkpoint weights/checkpoint_stage2_best.pt
```

### 批量图像推理

```bash
python main.py --mode inference --image_dir /path/to/image_folder --checkpoint weights/checkpoint_stage2_best.pt
```

批量推理结果将保存到 `outputs/inference_results.json`

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | inference | 运行模式：train/val/inference |
| `--image_path` | str | None | 单张图像推理路径 |
| `--image_dir` | str | None | 批量图像目录 |
| `--checkpoint` | str | None | 模型检查点路径 |
| `--train_stage` | int | 1 | 训练阶段：1/2/3 |
| `--batch_size` | int | 8 | 批次大小 |
| `--num_epochs` | int | 10 | 训练轮数 |
| `--lr` | float | 1e-4 | 学习率 |

## 训练阶段说明

### 阶段 1：特征对齐与路由预训练
- 冻结：CLIP 语义流、LLM 主干
- 训练：浅层 CNN 伪影流、MoME 融合网络
- 损失：BCE Loss（检测）+ DICE Loss（定位）+ CLM Loss（文本对齐）
- 目标：让视觉特征学会翻译成 LLM 能理解的语言

### 阶段 2：基于专家解释的深度指令微调
- 冻结：CLIP 语义流
- 训练：LLM LoRA、极简浅层 CNN 伪影流、MoME 融合网络
- 损失：加权交叉熵损失（对关键取证维度赋予更高权重）
- LoRA 配置：rank=64, alpha=32
- 目标：深化 LLM 对多维度取证维度的逻辑推理能力

## 评估指标

### 检测定位客观指标
- **AUC-ROC**: 曲线下面积
- **EER**: 等错误率
- **F1-Score**: F1 分数
- **mAP**: 平均精度
- **IoU**: 交并比（掩码定位）

### 解释质量文本指标
- **ROUGE-L**: 最长公共子序列
- **CIDEr**: 共识图像描述评估
- **LLM-as-a-Judge**: 预留接口（事实准确性、逻辑一致性）

## Trae AI 云端运行注意事项

1. **GPU 资源**：确保分配足够的 GPU 显存（建议 &gt;= 16GB）
2. **数据挂载**：将数据集挂载到 `data/` 目录
3. **检查点保存**：训练过程中模型会自动保存到 `weights/` 目录
4. **日志监控**：运行日志会保存到 `logs/` 目录
5. **输出查看**：验证和推理结果会保存到 `outputs/` 目录

## 核心模块说明

### 1. 双流多尺度视觉特征提取模块
- **语义流 (Semantic Stream)**: 使用 CLIP (ViT-L/14) 提取语义特征，支持提取中间层特征
- **底层伪影流 (Artifact Stream)**: 使用固定的 SRM 高通滤波器组 + 极浅层轻量级 CNN 提取伪影特征
  - SRM 滤波器：固定权重，强行剥离宏观语义，生成纯粹的噪声残差图
  - 极浅层 CNN：仅 4 层，提取高频统计特征

### 2. 视觉驱动的动态专家混合融合网络 (MoME)
- 构建专家池：语义几何专家、底层纹理专家、综合光影专家
- 动态软路由门控网络：根据图像特征自动分配专家权重
- 输出 Visual Forensic Tokens 作为 Soft Prompt 注入 LLM
- 纯视觉驱动，无需文本引导

### 3. LLM 推理核心
- 采用早期融合策略拼接视觉和文本令牌
- 集成 LoRA 注入接口用于参数高效微调
- 支持自然语言解释生成

### 4. 两阶段训练器
- 支持两个训练阶段的灵活切换
- 实现断点续训功能
- 自动保存最优模型
- 集成指标可视化

### 5. 多维度指标计算器
- 计算检测、定位、文本多维度指标
- 生成可视化图表（ROC、PR、柱状图）
- 预留 LLM-as-a-Judge 接口
