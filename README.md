
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
│   ├── forensic_cross_attention.py  # 取证感知交叉注意力适配器
│   ├── llm_infer.py       # LLM 推理核心
│   └── plaa_mllm.py       # PLAA-MLLM 整体组装
├── utils/                # 工具类目录
│   ├── device_utils.py   # 设备管理工具
│   ├── log_utils.py      # 日志工具
│   └── token_utils.py    # 分词工具
├── data/                 # 数据集目录（空）
├── weights/              # 模型权重目录（空）
├── outputs/              # 输出结果目录（空）
├── logs/                 # 日志目录（空）
├── requirements.txt      # 环境依赖清单
└── README.md             # 项目说明文档
```

## 文件调用关系

1. **main.py** - 项目入口，解析命令行参数并初始化各模块
2. **configs/** - 所有配置文件被其他模块导入使用
3. **models/plaa_mllm.py** - 核心组装类，整合三个主要模块：
   - `dual_stream_encoder.py` - 双流编码器
   - `forensic_cross_attention.py` - 交叉注意力适配器
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

### 推理模式

```bash
python main.py --mode inference --image_path /path/to/image.jpg
```

### 训练模式

```bash
python main.py --mode train
```

### 验证模式

```bash
python main.py --mode val --checkpoint /path/to/checkpoint.pt
```

## 核心模块说明

### 1. 双流多尺度视觉特征提取模块
- **语义流 (Semantic Stream)**: 使用 CLIP (ViT-L/14) 提取语义特征，支持提取中间层特征
- **底层伪影流 (Artifact Stream)**: 使用 ResNet + FPN 结构提取伪影特征

### 2. 取证感知交叉注意力适配器
- 使用隐式查询向量 (Latent Queries) 进行特征对齐
- 基于 Perceiver Resampler 实现交叉注意力计算
- 支持文本引导的动态特征提取

### 3. LLM 推理核心
- 采用早期融合策略拼接视觉和文本令牌
- 预留 LoRA 注入接口用于参数高效微调
