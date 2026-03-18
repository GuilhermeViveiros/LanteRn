#!/bin/bash

MODELS_TO_TEST=(
    # sft 4 lambda 0.1
    /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-1062
    # grpo lt 8 lambda 0.1
    #/mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-500
    #/mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1000
    #/mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1500
    # ntp lt 4
    #/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/checkpoint-1000
    #/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/checkpoint-5000
    #/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/checkpoint-10000
    #/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/checkpoint-16153
)

benchmarks=(
    "vstar_eval.py"
    "eval.py"
    "blink_eval.py"
)

# benchmarks=(
#     "eval.py"
# )

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
    python -m evals.eval --model_ref "$model" $use_lvr --batch_size 16 --output_dir "results/corrupted/empty_latent/"
    python -m evals.vstar_eval --model_ref "$model" $use_lvr --batch_size 16 --output_dir "results/corrupted/empty_latent/"
    python -m evals.blink_eval --model_ref "$model" $use_lvr --batch_size 16 --output_dir "results/corrupted/empty_latent/"


#         sbatch <<EOF
# #!/bin/bash
# #SBATCH --partition=a6000
# #SBATCH --qos=gpu-short
# #SBATCH --job-name=${job_name}_mc_high_duration
# #SBATCH --time=01:30:00
# #SBATCH --gpus=1
# #SBATCH --mem=120GB
# #SBATCH --cpus-per-task=40

# python evals/eval.py --model_ref "$model" $use_lvr --batch_size 16 --output_dir "results/corrupted/empty_latent/"
# EOF
#     done

done
