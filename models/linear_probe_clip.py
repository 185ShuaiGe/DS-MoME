import torch
import torch.nn as nn
from models.dual_stream_encoder import SemanticStream

class LinearProbeCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 复用你写好的语义流 (CLIP ViT-L/14)
        self.semantic_stream = SemanticStream(config)
        
        # 2. 核心：冻结 CLIP 的所有参数，使其不参与梯度更新
        for param in self.semantic_stream.parameters():
            param.requires_grad = False
            
        # 3. 定义线性探测头
        # 提取的是 layer_24 的 CLS token，CLIP ViT-L/14 的特征维度为 1024
        self.classifier = nn.Linear(1024, 1)
        
    def forward(self, image, text_prompt=None, text_guidance=None):
        # 主干网络的前向传播不计算梯度，极大地节省显存并加快速度
        with torch.no_grad(): 
            features = self.semantic_stream(image)
            # 提取最后一层 (layer_24) 序列中索引为 0 的 CLS token
            cls_token = features['layer_24'][:, 0, :] 
            
        # 只有这个线性分类头参与计算图和梯度更新
        logits = self.classifier(cls_token)
        return {'detection_logits': logits}