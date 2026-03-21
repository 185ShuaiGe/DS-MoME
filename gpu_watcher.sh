#!/bin/bash

# 接收用户输入的参数
INTERVAL=$1
COMMAND=$2

# 校验参数输入
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <等待时间t(分钟)> <'你的运行命令'>"
    echo "示例: $0 5 'python train.py --gpu_id'"
    exit 1
fi

# 5GB 显存阈值，转换为 MB (5 * 1024 = 5120)
THRESHOLD_MB=5120

while true; do
    # 使用 nvidia-smi 提取干净的 GPU 索引和已用显存数据 (格式: "0, 1024")
    gpu_status=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    # 逐行读取显卡状态
    while IFS=',' read -r gpu_index vram_used; do
        # 清理多余的空格
        gpu_index=$(echo "$gpu_index" | tr -d ' ')
        vram_used=$(echo "$vram_used" | tr -d ' ')

        # 判断当前显卡显存是否小于 5GB
        if [ "$vram_used" -lt "$THRESHOLD_MB" ]; then
            current_time=$(date "+%Y-%m-%d %H:%M:%S")
            
            # 按要求：在真正运行前输出唯一提示信息
            echo "[$current_time] 开始执行任务，使用的是显卡 GPU_ID: $gpu_index"
            
            # 执行自定义命令，并将 gpu_index 作为命令行参数追加在最后
            $COMMAND $gpu_index
            
            # 任务启动后，退出当前监控脚本 (避免重复提交相同任务)
            exit 0
        fi
    done <<< "$gpu_status"

    # 如果两张卡（4090和3090Ti）都在忙，静默等待 t 分钟后再次检查
    sleep $((INTERVAL * 60))
done