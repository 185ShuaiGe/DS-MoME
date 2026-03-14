import os
import shutil
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 1. 路径与配置
# ==========================================
CLEAN_TEST_ROOT = '/data/Disk_A/wangxinchang/Datasets/fdmas/test'
OUTPUT_BASE_DIR = '/data/Disk_A/wangxinchang/Datasets/fdmas'

# 定义 6 种扰动类型及参数
# 格式: '文件夹后缀': ('操作类型', 参数)
PERTURBATIONS = {
    'jpeg_90': ('jpeg', 90),
    'jpeg_75': ('jpeg', 75),
    'jpeg_50': ('jpeg', 50),
    'blur_1.0': ('blur', 1.0),
    'blur_2.0': ('blur', 2.0),
    'blur_3.0': ('blur', 3.0),
}

# ==========================================
# 2. 图像处理核心函数 (支持多进程)
# ==========================================
def process_single_image(task):
    src_path, dst_path, op_type, param = task
    
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    try:
        # 打开图像并确保为 RGB 模式
        img = Image.open(src_path).convert('RGB')
        
        if op_type == 'jpeg':
            # JPEG 压缩，强制保存为 JPEG 格式
            dst_path = os.path.splitext(dst_path)[0] + '.jpg'
            img.save(dst_path, 'JPEG', quality=param)
        
        elif op_type == 'blur':
            # 高斯模糊 (PIL 的 radius 类似于标准的 sigma)
            img = img.filter(ImageFilter.GaussianBlur(radius=param))
            img.save(dst_path)
            
        return True
    except Exception as e:
        return f"Error processing {src_path}: {e}"

# ==========================================
# 3. 主调度函数
# ==========================================
def main():
    print(f"🚀 开始生成鲁棒性测试数据集...")
    tasks = []

    # 遍历原图目录，收集所有需要处理的任务
    for root, dirs, files in os.walk(CLEAN_TEST_ROOT):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                src_path = os.path.join(root, file)
                
                # 计算相对路径 (例如: category_1/0_real/img1.jpg)
                rel_path = os.path.relpath(src_path, CLEAN_TEST_ROOT)
                
                # 为每种扰动生成一个任务
                for suffix, (op_type, param) in PERTURBATIONS.items():
                    # 目标根目录例如: /.../Datasets/fdmas/test_jpeg_90
                    target_dataset_root = os.path.join(OUTPUT_BASE_DIR, f'test_{suffix}')
                    dst_path = os.path.join(target_dataset_root, rel_path)
                    
                    tasks.append((src_path, dst_path, op_type, param))

    print(f"📦 共收集到 {len(tasks)} 个图像处理任务，开始多进程加速处理...")

    # 使用多进程加速 (根据 CPU 核心数自动分配)
    success_count = 0
    error_list = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            result = future.result()
            if result is True:
                success_count += 1
            else:
                error_list.append(result)

    print(f"\n✅ 数据集生成完毕！成功: {success_count} 张, 失败: {len(error_list)} 张")
    if error_list:
        print("⚠️ 部分失败日志:")
        for err in error_list[:10]: # 只打印前10个错误
            print(err)

if __name__ == '__main__':
    main()