import sys
import os

def count_image_files(folder_path):
    """
    统计指定文件夹下的图片文件数量
    
    Args:
        folder_path (str): 目标文件夹路径
    
    Returns:
        int: 图片文件数量
    """
    # 定义常见的图片文件扩展名（包含大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
                        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.WEBP'}
    
    image_count = 0
    
    # 遍历指定路径下的所有文件（不递归子文件夹）
    try:
        for item in os.listdir(folder_path):
            # 拼接完整路径
            item_path = os.path.join(folder_path, item)
            # 只统计文件（排除文件夹）
            if os.path.isfile(item_path):
                # 获取文件扩展名并判断是否为图片
                file_ext = os.path.splitext(item)[1]
                if file_ext in image_extensions:
                    image_count += 1
    except PermissionError:
        print(f"错误：没有访问 '{folder_path}' 的权限")
        sys.exit(1)
    except FileNotFoundError:
        print(f"错误：路径 '{folder_path}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误：{str(e)}")
        sys.exit(1)
    
    return image_count

def main():
    # 检查命令行参数数量
    if len(sys.argv) != 2:
        print("用法：python count_images.py <目标文件夹路径>")
        print("示例：python count_images.py /home/user/photos")
        sys.exit(1)
    
    # 获取命令行参数中的路径
    target_path = sys.argv[1]
    
    # 检查路径是否为目录
    if not os.path.isdir(target_path):
        print(f"错误：'{target_path}' 不是一个有效的文件夹路径")
        sys.exit(1)
    
    # 统计图片数量
    total_images = count_image_files(target_path)
    
    # 输出结果
    print(f"路径 '{target_path}' 下的图片数量：{total_images} 张")

if __name__ == "__main__":
    main()