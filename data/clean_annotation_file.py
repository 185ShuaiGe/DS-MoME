import json
import os

def clean_annotation_file(input_file, output_file):
    """
    清洗标注文件，删除label和expert_explanation矛盾的样本
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出清洗后的JSON文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return
    
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据结构是否包含samples字段
        if "samples" not in data:
            print("错误：JSON文件中未找到'samples'字段！")
            return
        
        original_samples = data["samples"]
        cleaned_samples = []
        deleted_count = 0
        
        # 遍历并过滤样本
        for idx, sample in enumerate(original_samples):
            # 确保样本包含必要的字段
            if "label" not in sample or "expert_explanation" not in sample:
                print(f"警告：第{idx}条样本缺少label或expert_explanation字段，已跳过")
                continue
            
            label = sample["label"]
            explanation = sample["expert_explanation"]
            
            # 转换为字符串并去除首尾空白，避免格式问题影响判断
            explanation_str = str(explanation).strip()
            
            # 判断是否需要删除该样本
            delete_flag = False
            # 条件1: label=0 但解释包含"This is a fake image"
            if label == 0 and "This is a fake image" in explanation_str:
                delete_flag = True
            # 条件2: label=1 但解释包含"real image"
            elif label == 1 and "real image" in explanation_str:
                delete_flag = True
            
            if delete_flag:
                deleted_count += 1
            else:
                cleaned_samples.append(sample)
        
        # 更新数据中的samples为清洗后的列表
        data["samples"] = cleaned_samples
        
        # 保存清洗后的文件（使用ensure_ascii=False保留中文，indent格式化输出）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 输出统计信息
        print(f"清洗完成！")
        print(f"原始样本数量：{len(original_samples)}")
        print(f"删除的矛盾样本数量：{deleted_count}")
        print(f"剩余样本数量：{len(cleaned_samples)}")
        print(f"清洗后的文件已保存至：{output_file}")
    
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的JSON文件！")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")

# 主程序入口
if __name__ == "__main__":
    # 配置文件路径（可根据你的实际路径修改）
    INPUT_FILE = "val/annotations_unified.json"
    OUTPUT_FILE = "val/annotations_cleaned.json"  # 输出新文件，避免覆盖原始数据
    
    # 执行清洗
    clean_annotation_file(INPUT_FILE, OUTPUT_FILE)