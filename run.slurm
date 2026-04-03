#!/bin/bash
#SBATCH -J AgentCode              # 作业名称
#SBATCH -o logs/Agent_%j.out            # 标准输出文件（%j = 作业ID）
#SBATCH -e logs/Agent_%j.err            # 标准错误文件
#SBATCH -p A100                      # 分区名称（可用 A800 / RTX4090 / L40 / A100）
#SBATCH --qos=normal                 # 服务质量
#SBATCH -N 1                         # 申请1个节点
#SBATCH -n 1                         # 申请1个任务（通常1个任务即可）
#SBATCH --cpus-per-task=8            # 每个任务分配8个CPU核心（根据需求调整）
#SBATCH --gres=gpu:8                 # 申请8块GPU
#SBATCH --mem=64G                    # 申请64GB内存（根据模型大小调整）

# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_TRACE_FUNCTION=1

# 加载所需模块
module load cuda/12.9.1              # 根据可用版本选择，也可用 cuda/13.0
module load Anaconda3/2025.06

# 激活 AlphaRL 环境（如果你已创建）
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate AlphaRL
# export CUDA_VISIBLE_DEVICES=3,4,5,6
nvidia-smi

# 切换到项目目录
cd /data2/group_王翔/xjh/projects/ReproAlphaRL/eval   # 请替换为你的实际路径

# 运行测试命令（选择一项或多项）

echo "========== Running Model Evaluation =========="
# bash dapo32b_eval.sh --model_keys rank_1pct --rank1pct_model_path /data2/group_王翔/xjh/projects/AgentAlphaRL/dapo32b/approx_top1
python dapo32b_eval.py \
    --model_keys approx_top1 \
    --custom_models "approx_top1:/data2/group_王翔/xjh/projects/AgentAlphaRL/dapo32b/approx_top1" \

# echo "========== Running =========="
# bash run_dapo32b_top1.sh



echo "========== Job Finished =========="
