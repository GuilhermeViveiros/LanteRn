#!/bin/bash

#conda activate lantern

# model configs
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LantErn-SFT"
REPO="/home/gviveiros/LantErn"
#export WANDB_DIR="/mnt/scratch-artemis/gviveiros/lantern/"

# dont use wandb for now
#export WANDB_DISABLED=True

RANDOM_SEED=42
DATA_PATH="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"

GLOBAL_BATCH_SIZE=120
BATCH_PER_DEVICE=6
NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_DEVICES"
# must be a multiple of BATCH_PER_DEVICE
if [ $((GLOBAL_BATCH_SIZE % BATCH_PER_DEVICE)) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be a multiple of BATCH_PER_DEVICE"
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Batch per device: $BATCH_PER_DEVICE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# LLM-related params
LR=1e-5
LVR_HEAD=False

# LantErn-related params
LANTERN_LOSS_FCT=mse

LAMBDA_LANTERN=0.1
LATENT_SIZE=8
CORRUPT_IMAGE=True
CORRUPTION_TYPE="bbox_blackout"
RUN_NAME="sft_${LANTERN_LOSS_FCT}_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}_corrupt_${CORRUPTION_TYPE}"

export OMP_NUM_THREADS=1
export PYTHONPATH=/home/gviveiros/LantErn:$PYTHONPATH

#--data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json \
deepspeed $REPO/src/train/train_sft.py \
    --deepspeed scripts/zero3.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 1 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /mnt/scratch-artemis/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to wandb \
    --resume_from_checkpoint True \
    --corrupt_image $CORRUPT_IMAGE \
    --corruption_type $CORRUPTION_TYPE