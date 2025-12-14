#!/bin/bash

MODELS_TO_TEST=(
    # lt_4_lambda_0.2
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.2/checkpoint-200"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.2/checkpoint-600"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.2/checkpoint-800"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.2/checkpoint-1062"
    # # lt_4_lambda_0.1
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-200"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-600"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-800"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-1062"
    # # lt_8_lambda_0.1
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-200"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-600"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-800"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062"
    # # lt_16_lambda_0.1
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_16_lambda_0.1/checkpoint-200"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_16_lambda_0.1/checkpoint-600"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_16_lambda_0.1/checkpoint-800"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_16_lambda_0.1/checkpoint-1062"
    # # lt_32_lambda_0.1
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_32_lambda_0.1/checkpoint-200"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_32_lambda_0.1/checkpoint-600"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_32_lambda_0.1/checkpoint-800"
    "/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_32_lambda_0.1/checkpoint-1062"
    # # next_token_prediction only
    "/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-200"
    "/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-600"
    "/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995"
    # qwen2.5-vl-3b-instruct
    Qwen/Qwen2.5-VL-3B-Instruct
    # qwen2.5-vl-7b-instruct
    Qwen/Qwen2.5-VL-7B-Instruct
)

benchmarks=(
    "vstar_eval.py"
    "viscot_eval.py"
    "blink_eval.py"
)

# iterate over the models and run the evaluation
for model in "${MODELS_TO_TEST[@]}"; do
    # Extract the job name as the checkpoint's trailing path, e.g. sft_mse_lt__lambda_0.1/checkpoint-995
    if [[ "$model" == *"Qwen"* ]]; then
        job_name="qwen"
    else
        job_name=$(echo "$model" | awk -F 'checkpoints/' '{print $2}')
    fi
    use_lvr="--lvr"
    echo "Evaluating model: $model on $job_name"
    # if model path contains Nuno or Qwen, use the following arg use_lvr=False
    if [[ "$model" == *"nuno"* || "$model" == *"Qwen"* ]]; then
        use_lvr="--no-lvr"
    fi
    for benchmark in "${benchmarks[@]}"; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --qos=gpu-short
#SBATCH --job-name=${job_name}_${benchmark}
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=40

python evals/${benchmark} --model_ref "$model" $use_lvr --batch_size 6
EOF
    done

done