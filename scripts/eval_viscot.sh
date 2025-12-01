#!/bin/bash

#SBATCH --job-name=viscot_eval
#SBATCH --output=logs/eval_viscot_%j.out
#SBATCH --error=logs/eval_viscot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=a6000
#SBATCH --qos=gpu-short
#SBATCH --time=03:00:00

# Activate conda environment if needed
#conda activate lantern

# Change to the project directory
cd /mnt/home/gviveiros/LantErn

# Model and data paths
# MODEL_REF can be passed as first argument, otherwise use default
MODEL_FOLDER="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/"
#MODEL_REF="${1:-warmup_01/checkpoint-100}"
#MODEL_REF="${1:-warmup_01/checkpoint-200}"
MODEL_REF="${1:-warmup_01/checkpoint-300}"
MODEL_PATH="${MODEL_FOLDER}/${MODEL_REF}"
DATA_PATH="${2:-/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json}"


echo "Model ref: $MODEL_REF"
echo "Data path: $DATA_PATH"


# Run the test script
srun python -m src.test \
    --model_ref "$MODEL_PATH" \
    --data_path "$DATA_PATH"

