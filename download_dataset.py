import os
# 核心配置：将默认端点替换为国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download



# 开启 hf_transfer 可以大幅提升下载速度（依赖于上一步 pip install hf_transfer）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 设置下载路径（指向你项目中的 data 目录）
local_download_dir = "./data/holmes_dataset"

print(f"开始从镜像站 (hf-mirror.com) 下载 AIGI-Holmes-Dataset...")
print(f"文件将保存在: {os.path.abspath(local_download_dir)}")

try:
    snapshot_download(
        repo_id="zzy0123/AIGI-Holmes-Dataset", 
        repo_type="dataset",
        local_dir=local_download_dir,
        resume_download=True,  # 开启断点续传（如果中断，重新运行脚本即可）
        max_workers=8,          # 启用多线程并发下载
        ignore_patterns=["TestSet.zip"]
    )
    print("\n🎉 数据集下载完成！")
except Exception as e:
    print(f"\n❌ 下载过程中出现错误: {e}")
    print("提示: 如果是因为网络波动中断，直接再次运行此脚本即可触发断点续传。")