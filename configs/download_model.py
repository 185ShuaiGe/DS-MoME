import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# ===================== 核心配置 =====================
# 模型名称
MODEL_ID = "openai/clip-vit-large-patch14"
# 指定保存路径 local
# SAVE_DIR = "D:\\cache\\huggingface_cache\\hub\\models--openai--clip-vit-large-patch14"
# autodl
SAVE_DIR = "/root/autodl-tmp/models/clip-vit-large-patch14"


# 国内Hugging Face镜像站（优先使用）
HF_MIRROR = "https://hf-mirror.com"

# ===================== 下载逻辑 =====================
def download_hf_model():
    # 1. 设置环境变量，优先使用镜像站
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"✅ 已设置镜像源：{HF_MIRROR}")
    
    # 2. 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✅ 保存目录已就绪：{SAVE_DIR}")

    try:
        # 3. 下载模型（核心函数）
        print(f"🚀 开始下载模型：{MODEL_ID}")
        snapshot_download(
            repo_id=MODEL_ID,          # 模型ID
            local_dir=SAVE_DIR,        # 保存路径
            local_dir_use_symlinks=False,  # Windows下禁用符号链接（避免权限问题）
            resume_download=True,      # 支持断点续传
            ignore_patterns=["*.git*"] # 忽略无关文件
        )
        print(f"🎉 模型下载完成！所有文件已保存至：{SAVE_DIR}")

    except RepositoryNotFoundError:
        print(f"❌ 错误：模型 {MODEL_ID} 不存在，请检查模型名称是否正确")
    except HfHubHTTPError as e:
        print(f"❌ 网络错误：{e}，请检查网络或镜像源是否可用")
    except PermissionError:
        print(f"❌ 权限错误：无法写入目录 {SAVE_DIR}，请以管理员身份运行脚本")
    except Exception as e:
        print(f"❌ 未知错误：{e}")

if __name__ == "__main__":
    download_hf_model()