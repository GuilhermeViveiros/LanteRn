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

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=8
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


RUN_NAME="Stage1_${LVR_LOSS_FCT}LVRLossLambda${LAMBDA_LVR}"
# ONLINE=True to enable online checkpointing with OCI
OUTPUT_DIR="stage1_checkpoints/"
LR=1e-5

# if continue training, set checkpoint_name = checkpoint to continue;
# --checkpoint_name checkpoint-1400

#deepspeed --num_gpus 2 --module src.train.train \

#DEEPSPEED=scripts/zero3.json

export OMP_NUM_THREADS=1
export PYTHONPATH=/home/gviveiros/LantErn:$PYTHONPATH

deepspeed $REPO/src/train/train.py \
    --run_name "Stage1_${LVR_LOSS_FCT}LVRLossLambda${LAMBDA_LVR}" \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_ID \
    --num_train_epochs 1 \
    --latent_size 4 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json \
    --output_dir /mnt/scratch-artemis/gviveiros/lantern/checkpoints/model_stage1 \
    --dummy False \
    --learning_rate $LR \
    --report_to wandb \


# python -m src.train.train \
#     --run_name "$RUN_NAME" \
#     --model_id $MODEL_ID \
#     --num_train_epochs 1 \
#     --latent_size 4 \
#     --per_device_train_batch_size $BATCH_PER_DEVICE \
#     --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
#     --data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json \
#     --output_dir /mnt/data-artemis/gviveiros/lantern/checkpoints/model_stage1 \
#     --dummy False \
#     --learning_rate $LR \
#     --report_to wandb \


# python -m src.train.train \
#     --run_name "$RUN_NAME" \
#     --model_id $MODEL_ID \
#     --latent_size 4 \
#     --gamma $LAMBDA_LANTERN \
#     --data_path /mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json \
#     --output_dir /mnt/data-artemis/gviveiros/lantern/checkpoints/model_stage1 \
#     --dummy False



# deepspeed src/train/train_lvr.py \
#     --run_name "$RUN_NAME" \
#     --coconut True \
#     --loss_lvr_fct $LVR_LOSS_FCT\
#     --deepspeed scripts/zero3_offload.json \
#     --model_id $MODEL_NAME \
#     --data_path "$DATA_PATH" \
#     --remove_unused_columns False \
#     --lvr_head $LVR_HEAD \
#     --freeze_vision_tower True \
#     --freeze_merger True \
#     --freeze_llm False \
#     --max_steps $MAX_STEPS \
#     --learning_rate $LR \
#     --loss_lvr_lambda $LAMBDA_LVR \
#     --bf16 True \
#     --fp16 False \
#     --disable_flash_attn2 False \
#     --online_checkpoint $ONLINE \
#     --output_dir "$OUTPUT_DIR" \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $BATCH_PER_DEVICE \
#     --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 False \
#     --gradient_checkpointing True \
#     --report_to wandb \
#     --lazy_preprocess True \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 10 \
#     --dataloader_num_workers 8 \
#     --enable_data_packing $DATA_PACKING \
#     --max_packed_tokens $MAX_PACKED_TOKENS \
#     --random_seed $RANDOM_SEED \
#     --long_seq_threshold $LST \
#     --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
#     # save_total_limit is for local storage only, no limit for online checkpointing