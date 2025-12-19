#!/bin/bash

#conda activate lantern

# model configs
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LantErn-SFT"
REPO="/mnt/home/gviveiros/LantErn"
#export WANDB_DIR="/mnt/scratch-artemis/gviveiros/lantern/"

# dont use wandb for now
#export WANDB_DISABLED=True

RANDOM_SEED=42
DATA_PATH="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"

GLOBAL_BATCH_SIZE=2 #128
BATCH_PER_DEVICE=1 #4
NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# must be a multiple of BATCH_PER_DEVICE
if [ $((GLOBAL_BATCH_SIZE % BATCH_PER_DEVICE)) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be a multiple of BATCH_PER_DEVICE"
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Batch per device: $BATCH_PER_DEVICE"
echo "Number of GPUs: $NUM_DEVICES"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# LLM-related params
LR=1e-5
LVR_HEAD=False

# LantErn-related params
LANTERN_LOSS_FCT=mse



RUN_NAME="sft_${LANTERN_LOSS_FCT}_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}"
# ONLINE=True to enable online checkpointing with OCI
OUTPUT_DIR="stage1_checkpoints/"


# if continue training, set checkpoint_name = checkpoint to continue;
# --checkpoint_name checkpoint-1400


#DEEPSPEED=scripts/zero3.json

export OMP_NUM_THREADS=1
export PYTHONPATH=/home/gviveiros/LantErn:$PYTHONPATH

LATENT_SIZE=4
LAMBDA_LANTERN=0.1
RUN_NAME="sft_mse_lt_4_lambda_0.1"
deepspeed $REPO/src/train/train.py \
    --deepspeed scripts/zero3.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 1 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json \
    --output_dir /mnt/scratch-artemis/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --use_ckpt False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to none \


#python src/train/train.py --use_ckpt False --model_id Qwen/Qwen2.5-VL-3B-Instruct --latent_size 4 --gamma 0.1 --data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json --output_dir /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1 --dummy False --learning_rate 1e-5 --gamma 0.1 --report_to none