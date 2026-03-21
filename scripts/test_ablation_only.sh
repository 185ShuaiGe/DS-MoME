#!/bin/bash

# ==============================================================================
# 脚本名称: test_ablation_only.sh
# 功能描述: 独立测试已经训练完成的消融实验组别 (A, B, C1)
# ==============================================================================

# 1. 默认参数设置
GPU_ID=0

# 2. 解析 Shell 命令行参数 (支持传入 --gpu_id)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id) GPU_ID="$2"; shift ;;
        *) echo "⚠️ 未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 日志输出双重保险
exec > >(tee -a ./logs/test_finished_ablations_4.log) 2>&1

echo "================================================================="
echo "🚀 开始独立测试已完成的消融组别"
echo "🕒 开始时间: $(date "+%Y-%m-%d %H:%M:%S")"
echo "================================================================="

# 你已经训练好的组别
EXPERIMENTS=("B" "D")

# 存放权重的绝对路径
BASE_DIR="/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/weights"

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "-----------------------------------------------------------------"
    echo "▶️ 正在测试组别: [ ${EXP} ]"
    echo "-----------------------------------------------------------------"

    if [ "${EXP}" == "final" ]; then
        WEIGHT_DIR="${BASE_DIR}/final"
    else
        WEIGHT_DIR="${BASE_DIR}/ablation"
    fi

    # 核心修复：匹配 "${EXP}-" 前缀（例如 A-0314...pt）
    # 使用 ls -t 保证如果有多份同组别权重，永远取最新的一份
    WEIGHT_FILE=$(ls -t ${WEIGHT_DIR}/${EXP}-*.pt 2>/dev/null | head -n 1)

    # 兼容性处理：如果找不到带短划线的，尝试找带下划线的
    if [ -z "${WEIGHT_FILE}" ]; then
        WEIGHT_FILE=$(ls -t ${WEIGHT_DIR}/${EXP}_*.pt 2>/dev/null | head -n 1)
    fi

    if [ -z "${WEIGHT_FILE}" ]; then
        echo "⚠️ 警告：找不到 [ ${EXP} ] 组的权重文件，跳过该组别！"
        continue
    fi

    echo "🔍 成功定位权重文件: ${WEIGHT_FILE}"
    echo "⏳ 开始执行推理测试..."

    # 调用你的测试脚本
    python test_ds_fdmas.py \
        --checkpoint "${WEIGHT_FILE}" \
        --gpu_id ${GPU_ID}

    echo "✅ [ ${EXP} ] 组测试执行完毕！"
done

echo "================================================================="
echo "🎉 独立测试流水线执行完毕！"
echo "================================================================="