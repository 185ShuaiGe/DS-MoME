import sys
import json

def count_jsonl_elements(file_path):
    """
    统计JSONL文件中'rejected_response'键值为空/非空字符串的元素数量
    
    Args:
        file_path (str): JSONL文件路径
    
    Returns:
        tuple: (总有效元素数, 值为空字符串的数量, 值不为空字符串的数量)
    """
    total_count = 0
    empty_rejected = 0   # "rejected_response"值为空字符串的元素数
    non_empty_rejected = 0  # "rejected_response"值非空字符串的元素数
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 跳过空行
                line = line.strip()
                if not line:
                    continue
                
                # 尝试解析为JSON对象
                try:
                    json_obj = json.loads(line)
                    total_count += 1
                    
                    # 确保键存在（兼容个别异常情况），并检查值是否为空字符串
                    if "rejected_response" in json_obj:
                        # 严格判断值是否为""（空字符串）
                        if json_obj["rejected_response"] == "":
                            empty_rejected += 1
                        else:
                            non_empty_rejected += 1
                    else:
                        # 兼容个别无该键的情况，给出提示
                        print(f"警告：第 {line_num} 行缺少'rejected_response'键，已归类为异常行")
                
                except json.JSONDecodeError:
                    print(f"警告：第 {line_num} 行是无效的JSON格式，已跳过：{line[:100]}...")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        sys.exit(1)
    except PermissionError:
        print(f"错误：没有访问 '{file_path}' 的权限")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误：{str(e)}")
        sys.exit(1)
    
    return total_count, empty_rejected, non_empty_rejected

def main():
    # 检查命令行参数数量
    if len(sys.argv) != 2:
        print("用法：python count_jsonl.py <jsonl文件路径>")
        print("示例：python count_jsonl.py data.jsonl")
        sys.exit(1)
    
    # 获取文件路径
    jsonl_path = sys.argv[1]
    
    # 统计各类元素数量
    total, empty_rej, non_empty_rej = count_jsonl_elements(jsonl_path)
    
    # 输出结果
    print(f"==== JSONL文件统计结果 ====")
    print(f"文件路径：{jsonl_path}")
    print(f"总有效JSON元素数量：{total} 个")
    print(f"'rejected_response'值为空字符串（\"\"）的元素数量：{empty_rej} 个")
    print(f"'rejected_response'值不为空字符串的元素数量：{non_empty_rej} 个")
    
    # 校验总数（可选，方便排查异常）
    if empty_rej + non_empty_rej != total:
        print(f"\n⚠️  注意：空值+非空值数量（{empty_rej + non_empty_rej}）与总数（{total}）不一致，可能存在缺少'rejected_response'键的行")

if __name__ == "__main__":
    main()