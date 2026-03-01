import os
from huggingface_hub import snapshot_download

# 配置国内镜像站加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 开启 hf_transfer 极速下载
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
SAVE_DIR = "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.2"

def download_llm():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"🚀 开始下载 LLM 模型到：{SAVE_DIR}")
    
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=SAVE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8
        )
        print("🎉 Mistral-7B 模型下载完成！")
    except Exception as e:
        print(f"❌ 下载出错: {e}")

if __name__ == "__main__":
    download_llm()