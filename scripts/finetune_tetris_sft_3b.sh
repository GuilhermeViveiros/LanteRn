#!/bin/bash

# model configs
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LantErn-SFT"
REPO="/home/gviveiros/LantErn"

RANDOM_SEED=42
DATA_PATH="/mnt/scratch-nyx/gviveiros/lantern/analogy_data/train.json"

GLOBAL_BATCH_SIZE=160
BATCH_PER_DEVICE=10
NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_DEVICES"

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
export PYTHONPATH=/home/gviveiros/LantErn:$PYTHONPATH

deepspeed $REPO/src/train/train_sft.py \
    --deepspeed $REPO/scripts/zero2.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 10 \
    --max_train_samples 50000 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /mnt/scratch-nyx/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to wandb \
    --dataset_type tetris \
    --data_path $DATA_PATH \
    --eval_strategy steps \
    --eval_steps 200 \
    --per_device_eval_batch_size 8
