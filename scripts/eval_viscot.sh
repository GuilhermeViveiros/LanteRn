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
#SBATCH --time=00:20:00

# Activate conda environment if needed
#conda activate lantern

# Change to the project directory
cd /mnt/home/gviveiros/LantErn

# Get Model Path
MODEL_PATH="${1}"
OUTPUT_DIR="${2}"

# if the output directory is not provided, use the default
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="results"
fi

# check if the model path is valid
if [ -z "$MODEL_PATH" ]; then
    echo "Model path is required"
    exit 1
fi
 
DATA_PATH="/e/project1/jureap126/gviveiros/lantern/LantErn_VisCot_data.json"


echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"


# Run the test script
srun python -m src.test \
    --model_ref "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 12 \
    # --use_gt True

