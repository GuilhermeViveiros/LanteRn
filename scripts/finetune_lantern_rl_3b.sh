#!/bin/bash


# model configs
export WANDB_PROJECT="LantErn-GRPO"
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/gviveiros/LantErn:$PYTHONPATH

MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
REPO="/home/gviveiros/LantErn"
RANDOM_SEED=42

NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_DEVICES"

LATENT_SIZE=8
LAMBDA_LANTERN=0.1
RUN_NAME="grpo_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}"
CHECKPOINT_PATH="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}/checkpoint-1062"

deepspeed --num_gpus $NUM_DEVICES $REPO/src/train/train_grpo.py \
    --deepspeed scripts/zero3.json \
    --run_name $RUN_NAME \
    --model_path $CHECKPOINT_PATH \
    --output_dir /mnt/scratch-artemis/gviveiros/lantern/checkpoints/$RUN_NAME \
    --save_steps 200 \
    \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --beta 0.1 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 3 \
    \
    --per_device_train_batch_size 12 \
    --num_generations 6 \
    --max_completion_length 320 \
    --temperature 0.6 \
    --seed $RANDOM_SEED \
    --top_p 0.85 \
    \
    --logging_strategy steps \
    --logging_steps 50 \
    \
    --reward_names accuracy structure \
    --reward_weights 1.0 1.0 \
    \
    --data_path "/mnt/scratch-hades/nunogoncalves/LantErn/rl_dataset/lvr_data/virl39k.json" \
    --image_root "/mnt/data-hades/gviveiros/"\
    --report_to wandb