#!/bin/bash
# Submits one sbatch job per benchmark for Qwen2.5-VL-7B-Instruct (runs in parallel)
REPO="/e/home/jusers/viveiros1/jupiter/LantErn"

for benchmark in viscot blink vstar tetris; do
    sbatch \
        --account=jureap126 \
        --partition=booster \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --time=12:00:00 \
        --job-name="baseline-7b-${benchmark}" \
        --output="/e/project1/jureap126/gviveiros/lantern/logs/baseline_qwen_7b_${benchmark}_%j.out" \
        --error="/e/project1/jureap126/gviveiros/lantern/logs/baseline_qwen_7b_${benchmark}_%j.err" \
        --wrap="
            mkdir -p /e/project1/jureap126/gviveiros/lantern/logs
            export PYTHONPATH=${REPO}:\$PYTHONPATH
            python -m evals.baseline_qwen_eval \
                --model_ref Qwen/Qwen2.5-VL-7B-Instruct \
                --benchmarks ${benchmark} \
                --output_dir ${REPO}/results/baseline_qwen_7b
        "
    echo "Submitted baseline-7b-${benchmark}"
done
