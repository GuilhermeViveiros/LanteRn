#!/bin/bash
#SBATCH -A jureap126
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=filter-dummy
#SBATCH --output=results/filter_training/slurm-dummy-%j.out

set -e
cd /e/home/jusers/viveiros1/jupiter/LantErn
export HF_HUB_OFFLINE=1
export PYTHONPATH=/e/home/jusers/viveiros1/jupiter/LantErn:$PYTHONPATH

python -u -m synthetic.viscot.filter_training_samples \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --dummy 20 \
    --batch_size 2
