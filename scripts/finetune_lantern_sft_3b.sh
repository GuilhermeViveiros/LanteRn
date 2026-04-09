#!/bin/bash
#SBATCH --account=jureap126
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=lantern-sft-3b
#SBATCH --output=/e/project1/jureap126/gviveiros/lantern/logs/sft_3b.out
#SBATCH --error=/e/project1/jureap126/gviveiros/lantern/logs/sft_3b.err

# model configs
MODEL_ID="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
export WANDB_MODE=offline
export WANDB_PROJECT="LantErn-SFT"
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"

RANDOM_SEED=42
DATA_PATH="/e/project1/jureap126/gviveiros/lantern/LantErn_VisCot_data.json"
FILTER_IDS_PATH="$REPO/results/filter_training/keep_and_hard_ids.json"  # 59616 samples (keep + hard, excludes easy)

GPUS_PER_NODE=4
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=$(( SLURM_NNODES * GPUS_PER_NODE ))
echo "Nodes: $SLURM_NNODES, GPUs per node: $GPUS_PER_NODE, total devices: $NUM_DEVICES"

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

# LantErn-related params
LAMBDA_LANTERN=0.1
LATENT_SIZE=8
CORRUPT_IMAGE=False
CORRUPTION_TYPE="bbox_blackout"
FREEZE_LATENT_ONLY=False
LATENT_LOSS_TYPE="infonce"   # mse | infonce | cosine
TEMPERATURE=0.07
RUN_NAME="jupi_sft_${LATENT_LOSS_TYPE}_lt_${LATENT_SIZE}_lambda_${LAMBDA_LANTERN}"

export OMP_NUM_THREADS=1
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

mkdir -p /e/project1/jureap126/gviveiros/lantern/logs

# One srun task per node; torchrun spawns GPUS_PER_NODE processes per node.
# $SLURM_NODEID must be escaped so it is evaluated per-task, not in the batch script.
srun --ntasks-per-node=1 bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $REPO/src/train/train_sft.py \
    --deepspeed $REPO/scripts/zero2.json \
    --run_name $RUN_NAME \
    --model_id $MODEL_ID \
    --num_train_epochs 1 \
    --latent_size $LATENT_SIZE \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir /e/project1/jureap126/gviveiros/lantern/checkpoints/$RUN_NAME \
    --dummy False \
    --learning_rate $LR \
    --gamma $LAMBDA_LANTERN \
    --report_to none \
    --resume_from_checkpoint False \
    --corrupt_image $CORRUPT_IMAGE \
    --corruption_type $CORRUPTION_TYPE \
    --freeze_latent_only $FREEZE_LATENT_ONLY \
    --latent_loss_type $LATENT_LOSS_TYPE \
    --temperature $TEMPERATURE"
