#!/bin/bash
#SBATCH -A jureap126
#SBATCH -p booster
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --time=12:00:00
#SBATCH --job-name=filter-train
#SBATCH --output=results/filter_training/slurm-%j.out

set -e
cd /e/home/jusers/viveiros1/jupiter/LantErn
export HF_HUB_OFFLINE=1
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m synthetic.viscot.filter_training_samples \
    --batch_size 4
