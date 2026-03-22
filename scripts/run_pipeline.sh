#!/bin/bash

# ==============================================================================
# 脚本名称: run_pipeline.sh
# 存放路径: ./scripts/run_pipeline.sh
# 功能描述: 自动化执行 DS-MoME 的训练与 FDMAS 测试流水线
# ==============================================================================

# 无论从哪里执行脚本，自动获取脚本所在目录，并强制进入上一级的项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_ROOT}"

# ==================== 配置区 ====================
# 定义超参数（便于后续统一修改）
GPU_ID=$1
MODE="train"
BATCH_SIZE=4
NUM_EPOCHS=10
LR="1e-4"
ABLATION="final"

# 权重保存的绝对路径基础目录
WEIGHT_BASE_DIR="/data/Disk_A/wangxinchang/DeepfakeDetectionMethods/DS-MoME/weights"

# 创建日志目录并将终端输出双重记录到日志文件中
mkdir -p ./logs
LOG_FILE="./logs/run_pipeline_$(date "+%Y%m%d_%H%M%S").log"
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "================================================================="
echo "🚀 开始执行 DS-MoME 自动化训练与测试流水线"
echo "🕒 开始时间: $(date "+%Y-%m-%d %H:%M:%S")"
echo "📂 当前工作目录: $(pwd)"
echo "================================================================="

# ==================== 第一阶段：训练 ====================
echo "⏳ [1/2] 开始训练模型 (Ablation: ${ABLATION})..."

python main.py \
    --mode ${MODE} \
    --gpu_id ${GPU_ID} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --ablation ${ABLATION}

# 检查训练是否成功退出
if [ $? -ne 0 ]; then
    echo "❌ 训练过程中发生错误，自动化流水线已中止！"
    exit 1
fi
echo "✅ 训练成功完成！"

# ==================== 第二阶段：寻找最佳权重 ====================
# 根据 trainer.py 中的逻辑确定最终存放权重的子文件夹
if [ "${ABLATION}" == "final" ]; then
    WEIGHT_DIR="${WEIGHT_BASE_DIR}/final"
else
    WEIGHT_DIR="${WEIGHT_BASE_DIR}/ablation"
fi

# 使用 ls -t (按时间倒序) 和 head -n 1 提取最新生成的该组别的权重文件
# 匹配规则参考 trainer.py: {exp_id}-{timestamp}-{epochs}-{lr}.pt
BEST_WEIGHT=$(ls -t ${WEIGHT_DIR}/${ABLATION}-*.pt 2>/dev/null | head -n 1)

if [ -z "${BEST_WEIGHT}" ]; then
    echo "❌ 错误：在 ${WEIGHT_DIR} 下找不到以 ${ABLATION}- 开头的最优权重文件，无法进行测试！"
    exit 1
fi

echo "🔍 成功定位最新生成的最佳权重文件: ${BEST_WEIGHT}"

# ==================== 第三阶段：测试 ====================
echo "⏳ [2/2] 开始在 FDMAS 数据集上测试该权重..."

python test_ds_fdmas.py \
    --checkpoint "${BEST_WEIGHT}" \
    --gpu_id ${GPU_ID}

if [ $? -ne 0 ]; then
    echo "❌ 测试过程中发生错误！"
    exit 1
fi

echo "================================================================="
echo "🎉 训练与测试流水线全部执行完毕！"
echo "🕒 结束时间: $(date "+%Y-%m-%d %H:%M:%S")"
echo "📄 完整日志已保存至: ${LOG_FILE}"
echo "================================================================="