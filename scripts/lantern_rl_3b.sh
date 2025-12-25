#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="LantErn-SFT"
export WANDB_ENTITY="nuno-m-goncalves-ul"
export WANDB_START_METHOD=thread

# Important to set the rewards. On the rewards.py we can define different rewards and register them.
# The name of the registered rewards need to be passed on the command line below along with the weight of the reward.
# Example: 
# --reward_names accuracy lvr_presence 
# --reward_weights 1.0 1.0

python -m src.rl.run \
  \
  --model_path "/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995" \
  --freeze_vision True \
  --freeze_vision_proj False \
  \
  --output_dir "/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/" \
  --run_name "lantern-test" \
  --report_to wandb \
  \
  --learning_rate 5e-6 \
  --warmup_ratio 0.03 \
  --beta 0.1 \
  \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  \
  --num_generations 4 \
  --max_completion_length 128 \
  --temperature 0.6 \
  --top_p 0.85 \
  \
  --logging_steps 100 \
  \
  --reward_names accuracy lvr_presence \
  --reward_weights 1.0 1.0 \
  \
  --json_path "/mnt/scratch-hades/nunogoncalves/LantErn/rl_dataset/lvr_data/virl39k.json" \
  --image_root "/mnt/scratch-hades/nunogoncalves/LantErn/rl_dataset/"

