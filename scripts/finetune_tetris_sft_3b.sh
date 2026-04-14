#!/bin/bash
#SBATCH --account=jureap126
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=sft_tetris_ablation1_gray
#SBATCH --output=logs/sft_tetris_ablation1_gray.out
#SBATCH --error=logs/sft_tetris_ablation1_gray.err

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
NUM_DEVICES=$(( SLURM_NNODES * GPUS_PER_NODE ))
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
RUN_NAME="ablation1_gray_${LATENT_LOSS_TYPE}_lt${LATENT_SIZE}_lambda${LAMBDA_LANTERN}"

export OMP_NUM_THREADS=1
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
    --eval_steps 100 \
    --per_device_eval_batch_size 6 \
    --eval_accumulation_steps 2 \
    --latent_loss_type $LATENT_LOSS_TYPE \
    --grayscale_intermediate True \
    --wandb_project LantErn-Tetris
    #--temperature 0.07

# torchrun --nproc_per_node=$GPUS_PER_NODE $TRAIN_ARGS


# srun --ntasks-per-node=1 bash -c "torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --node_rank=\$SLURM_NODEID \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     $TRAIN_ARGS"


# python -m evals.generalization --checkpoints_dir /e/project1/jureap131/gviveiros/lantern/checkpoints/lantern_tetris_sft_lt_8_lambda_0.1/checkpoint-672 --held_out_path /e/project1/jureap131/gviveiros/lantern/analogy_data/held_out/held_out.json --eval_path /e/project1/jureap131/gviveiros/lantern/analogy_data/eval.json  --n_samples 200 --batch_size 8 --latent_size 8 --use_lvr

# python -m evals.generalization --checkpoints_dir /e/project1/jureap131/gviveiros/lantern/checkpoints/ntp_tetris_sft_3b/checkpoint-672/ --held_out_path /e/project1/jureap131/gviveiros/lantern/analogy_data/held_out/held_out.json --eval_path /e/project1/jureap131/gviveiros/lantern/analogy_data/eval.json  --n_samples 200 --batch_size 8 --no_lvr