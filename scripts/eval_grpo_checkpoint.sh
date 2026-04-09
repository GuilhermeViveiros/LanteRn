#!/bin/bash
# Submits one sbatch job per benchmark for the GRPO RL checkpoint (runs in parallel)
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"
CHECKPOINT="/e/project1/jureap126/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1500"

for benchmark in viscot blink vstar; do
    sbatch \
        --account=jureap126 \
        --partition=booster \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --time=12:00:00 \
        --job-name="grpo-1500-${benchmark}" \
        --output="/e/project1/jureap126/gviveiros/lantern/logs/grpo_1500_${benchmark}_%j.out" \
        --error="/e/project1/jureap126/gviveiros/lantern/logs/grpo_1500_${benchmark}_%j.err" \
        --wrap="
            mkdir -p /e/project1/jureap126/gviveiros/lantern/logs
            export PYTHONPATH=${REPO}:\$PYTHONPATH
            python -m evals.viscot_blink_vstar_eval \
                --model_ref ${CHECKPOINT} \
                --benchmarks ${benchmark} \
                --output_dir ${REPO}/results/grpo_1500
        "
    echo "Submitted grpo-1500-${benchmark}"
done
