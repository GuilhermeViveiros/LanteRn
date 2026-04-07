#!/bin/bash
#SBATCH --account=jureap131
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=sft_lantern_3b
#SBATCH --output=logs/sft_lantern_3b.out
#SBATCH --error=logs/sft_lantern_3b.err

# model configs
MODEL_ID="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
export WANDB_MODE=offline
export WANDB_PROJECT="LantErn-SFT"
export WANDB_DIR="/e/project1/jureap131/gviveiros/lantern/"
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"

RANDOM_SEED=42
DATA_PATH="/e/project1/jureap131/gviveiros/lantern/analogy_data/train.json"

GPUS_PER_NODE=4
GLOBAL_BATCH_SIZE=180
BATCH_PER_DEVICE=5
NUM_DEVICES=$(( SLURM_NNODES * GPUS_PER_NODE ))
echo "Nodes: $SLURM_NNODES, GPUs per node: $GPUS_PER_NODE, total devices: $NUM_DEVICES"

if [ $((GLOBAL_BATCH_SIZE % (BATCH_PER_DEVICE * NUM_DEVICES))) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be a multiple of BATCH_PER_DEVICE"
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Batch per device: $BATCH_PER_DEVICE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# LLM-related params
LR=1e-5

# LantErn-related params
LAMBDA_LANTERN=0.1
LATENT_SIZE=8
FREEZE_LATENT_ONLY=False
RUN_NAME="lantern_tetris_sft_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}"

export OMP_NUM_THREADS=1
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

mkdir -p /e/project1/jureap131/gviveiros/lantern/logs

TRAIN_ARGS="$REPO/src/train/train_sft.py \
    --deepspeed $REPO/scripts/zero2.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 2 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /e/project1/jureap131/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to wandb \
    --dataset_type tetris \
    --data_path $DATA_PATH \
    --eval_strategy steps \
    --eval_steps 20 \
    --per_device_eval_batch_size 8"


srun --ntasks-per-node=1 bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_ARGS"
