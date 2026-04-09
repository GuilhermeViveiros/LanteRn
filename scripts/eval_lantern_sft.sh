#!/bin/bash
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"
CHECKPOINT="/e/project1/jureap126/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/"

sbatch \
    --account=jureap126 \
    --partition=booster \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --gres=gpu:1 \
    --time=12:00:00 \
    --job-name="lantern-sft-eval" \
    --output="/e/project1/jureap126/gviveiros/lantern/logs/lantern_sft_eval_%j.out" \
    --error="/e/project1/jureap126/gviveiros/lantern/logs/lantern_sft_eval_%j.err" \
    --wrap="
        mkdir -p /e/project1/jureap126/gviveiros/lantern/logs
        export PYTHONPATH=${REPO}:\$PYTHONPATH
        python -m evals.viscot_blink_vstar_eval \
            --model_ref ${CHECKPOINT} \
            --benchmarks viscot \
            --output_dir ${REPO}/lan/lantern_sft/
    "
echo "Submitted lantern-sft-eval"
