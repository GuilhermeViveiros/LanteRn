#!/bin/bash
#SBATCH --account=jureap126
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=lantern-sft-7b
#SBATCH --output=/e/project1/jureap126/gviveiros/lantern/logs/sft_7b.out
#SBATCH --error=/e/project1/jureap126/gviveiros/lantern/logs/sft_7b.err

# model configs
MODEL_ID="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/$(ls $HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/ | head -1)"
export WANDB_MODE=offline
export WANDB_PROJECT="LantErn-SFT"
export WANDB_DIR="/e/project1/jureap126/gviveiros/lantern/"
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"

DATA_PATH="/e/project1/jureap126/gviveiros/lantern/LantErn_VisCot_data.json"

GPUS_PER_NODE=4
GLOBAL_BATCH_SIZE=192
BATCH_PER_DEVICE=3
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

# LLM-related params
LR=1e-5

# LantErn-related params
LAMBDA_LANTERN=0.1
LATENT_SIZE=8
LATENT_LOSS_TYPE="mse"   # mse | infonce | cosine
RUN_NAME="lantern_sft_7b_mse_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}"

export OMP_NUM_THREADS=1
source /e/project1/jureap126/gviveiros/envs/swift/bin/activate
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

mkdir -p /e/project1/jureap126/gviveiros/lantern/logs

deepspeed $REPO/src/train/train_sft.py \
    --deepspeed $REPO/scripts/zero3.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 1 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /e/project1/jureap126/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --resume_from_checkpoint False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to wandb \
    --dataset_type viscot \
    --data_path $DATA_PATH \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 3 \
    --eval_accumulation_steps 2 \
    --latent_loss_type $LATENT_LOSS_TYPE \
    --wandb_project LantErn-SFT
