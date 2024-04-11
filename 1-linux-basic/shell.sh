#!/bin/bash
# 获取主机的所有IP地址，然后从这些地址中获取并打印第一个IP地址
IP_ADDR=$(hostname -I | awk '{print $1}')
echo "IP address: $IP_ADDR"

# if command -v nvidia-smi &> /dev/null; then：这行代码使用 command -v 检查 nvidia-smi 命令是否存在。
# &> /dev/null 是将输出重定向到 /dev/null，其含义是忽略任何输出信息，即不会在终端中显示任何命令输出。
# GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)：
# 通过 nvidia-smi --query-gpu=name --format=csv,noheader 来查询 GPU 的名称，输出信息为 CSV 格式并且没有表头行。然后通过管道操作 |，将上述输出传递给 wc -l 命令计算 GPU 的数量（通过计算行数）。
# 最后将所得结果赋值给变量 GPU_COUNT。
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "NUM GPU: $GPU_COUNT"
else
    echo "nvidia-smi 命令不存在。请确保你安装了NVIDIA驱动。"
fi

# 用于创建HOSTFILE_NAME中的IP和slot个数
HOSTFILE_NAME="hostfile_$IP_ADDR"
echo "$IP_ADDR slots=$GPU_COUNT" > $HOSTFILE_NAME

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=$GPU_COUNT
HOST_FILE_PATH="$HOSTFILE_NAME"
OPTIONS_NCCL="NCCL_DEBUG=version NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_GID_INDEX=3 NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

echo $main_dir

MODEL_FILE="$1"
tasks=("$@")

MODEL_MODULE="models.glm_model.GLMModel"
# ADD_FILE="/workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/ptuning/output/sciglm-label10_mi2_mwp2-chatglm2-6b-ft-2e-5-0.1/checkpoint-12090"
MP_SIZE=1
DATA_PATH="/workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/evaluation/glm-evals-dev/datasets"
mkdir -p logs
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}


if [[ ! -d $MODEL_FILE ]]; then
    echo "The provided MODEL_FILE directory does not exist."
    exit 1
fi
counter=0
# 为文件夹内的所有checkpoint-{数字}文件进行循环
for checkpoint in $(ls -d $MODEL_FILE/checkpoint-* | sort -V); do
    # if [[ $counter -eq 1 ]]; then
    #     break
    # fi
    # ((counter++))
    MODEL_NAME=$checkpoint
    echo $MODEL_NAME
    # bash /workspace/xuyifan/xuyifan-old/glm-evals/configs/6b-math/fix_cpt.sh $MODEL_NAME $ADD_FILE

    MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --max-sequence-length 32768 \
            --tokenizer-type GLMSentencePieceTokenizer \
            --tokenizer-model-path ./tokenization/embed_assets/65k_tokenizer.model \
            --model-module ${MODEL_MODULE} \
            --model-name 6b \
            --prefix-type glm \
            --prompt-style english \
            --skip-init \
            --bf16"

    ARGS="${main_dir}/evaluate.py \
       --data-path $DATA_PATH \
       --save-dir ${MODEL_NAME}/${STEP}/evaluation \
       --load-hf-weight ${MODEL_NAME} \
       --task ${tasks[*]:1} \
       --overwrite-result \
       $MODEL_ARGS"

    run_cmd="${OPTIONS_NCCL} /workspace/zhangdan/miniconda3/bin/deepspeed --master_port 12347 --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${ARGS}"
    eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}_${MODEL_NAME##*/}.log

done

# run_cmd="${OPTIONS_NCCL} /workspace/zhangdan/miniconda3/bin/deepspeed --include="localhost:2,3,4,6" --master_port 12347 ${ARGS}"
# cd /workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/evaluation/glm-evals-dev
# bash generate_ssh.sh
# bash scripts/evaluate_each_ckpt.sh "/workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/ptuning/output/sciglm-label10_mi2_mwp2-chatglm2-6b-ft-2e-5-0.1" tasks/MATH/MATH.yaml

# conda activate chatglm2
# cd /workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/evaluation/glm-evals-dev
# bash generate_ssh.sh
# CUDA_VISIBLE_DEVICES=2,3,4,6 bash scripts/evaluate_each_ckpt.sh "/workspace/zhangdan/code/SciGLM/ChatGLM2-6B-main/ptuning/output/sciglm-6b/sciglm-label5-mic-mm-lean-chatglm2-6b-ft-2e-5-0.1-inverse_sqrt-epoch1.2" tasks/gsm8k/gsm8k_zero.yaml