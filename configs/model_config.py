
class ModelConfig:
    clip_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/cache/models--openai--clip-vit-large-patch14"  
    llm_model_name = "/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct"

    clip_dim = 1024             # CLIP 模型输出维度
    clip_intermediate_layers = [8, 16, 24]  # 选择 CLIP 的哪些层作为语义特征输入
    num_latent_queries = 256                # 生成的潜在查询数量，原始128
    latent_dim = 1024                   # 潜在查询的特征维度，原始512                
    
    llm_dim = 4096                  # LLM 模型输出维度
    grad_accum_steps = 4            # 梯度累积步数，原始8
    max_seq_len = 1024              # 最大序列长度
