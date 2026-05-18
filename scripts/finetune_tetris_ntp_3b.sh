#!/bin/bash
#SBATCH --account=jureap131
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=sft_tetris_ntp_small
#SBATCH --output=logs/sft_tetris_ntp_small.out
#SBATCH --error=logs/sft_tetris_ntp_small.err

# model configs
MODEL_ID="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
export WANDB_MODE=offline
export WANDB_PROJECT="LantErn-Tetris"
export WANDB_DIR="/e/project1/jureap131/gviveiros/lantern/"
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"


DATA_PATH="/e/project1/jureap131/gviveiros/lantern/analogy_data/train.json"

GPUS_PER_NODE=4
GLOBAL_BATCH_SIZE=192
BATCH_PER_DEVICE=6
if [[ -z "${SLURM_NNODES}" ]]; then
  NUM_DEVICES=1
else
  NUM_DEVICES=$(( SLURM_NNODES * GPUS_PER_NODE ))
fi
echo "Nodes: $SLURM_NNODES, GPUs per node: $GPUS_PER_NODE, total devices: $NUM_DEVICES"

GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Batch per device: $BATCH_PER_DEVICE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

LR=1e-5
RUN_NAME="sft_tetris_ntp_small"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /e/project1/jureap126/gviveiros/envs/swift/bin/activate
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

mkdir -p /e/project1/jureap131/gviveiros/lantern/logs

deepspeed $REPO/src/train/train_sft.py \
    --deepspeed $REPO/scripts/zero2.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 20 \
    --latent_size 8 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /e/project1/jureap131/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --max_train_samples 4000 \
    --resume_from_checkpoint False \
    --learning_rate $LR \
    --gamma 0.0 \
    --use_lvr False \
    --report_to wandb \
    --dataset_type tetris \
    --data_path $DATA_PATH \
    --eval_strategy steps \
    --eval_steps 80 \
    --per_device_eval_batch_size 6 \
    --eval_accumulation_steps 2 \
    --wandb_project LantErn-Tetris
