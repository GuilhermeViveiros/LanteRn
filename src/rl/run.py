import os

# ------------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------------
import torch
from datasets import load_dataset
from datasets.features import Image as HFImage
from datasets.features import Sequence
from transformers import HfArgumentParser
from trl import GRPOTrainer

# ------------------------------------------------r------------------
# Local application imports
# ------------------------------------------------------------------
from src.models import load_model

from src.rl.config import GRPOParams, ImageDataset
from src.rl.prompt import build_system_prompt
from src.rl.utils import configure_generation_cache, convert_example, freeze_vision


from src.rl import rewards as rewards_module


def build_reward_funcs(grpo_params: GRPOParams, model):  # noqa: F811
    names = grpo_params.reward_names

    missing = [n for n in names if n not in rewards_module.REWARD_REGISTRY]
    if missing:
        available = ", ".join(sorted(rewards_module.REWARD_REGISTRY.keys()))
        raise ValueError(f"Unknown reward(s): {missing}. Available: [{available}]")

    reward_funcs = [rewards_module.REWARD_REGISTRY[n](model) for n in names]

    if len(grpo_params.reward_weights) != len(reward_funcs):
        raise ValueError(
            f"reward_weights length ({len(grpo_params.reward_weights)}) must match "
            f"reward_names length ({len(reward_funcs)}). "
            f"names={grpo_params.reward_names} weights={grpo_params.reward_weights}"
        )

    return reward_funcs


def main(grpo_params: GRPOParams, json_path: str, image_root: str):
    # ----------------------------
    # 0) Prepare model and tokenizer
    # ----------------------------

    model, processor = load_model(
        # model_path="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995",
        model_path=grpo_params.model_path,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        compute_dtype=torch.bfloat16,
        use_cache=True,
    )

    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
       processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ensure_lvr_tokens(model, processor)
    configure_generation_cache(model)
    if grpo_params.freeze_vision:
        freeze_vision(model, freeze_projector=grpo_params.freeze_vision_proj)

    # ----------------------------
    # 1) Prepare rewards
    # ----------------------------
    reward_funcs = build_reward_funcs(grpo_params, model)

    # ----------------------------
    # 2) Load dataset
    # ----------------------------

    latent_size = int(model.config.latent_size)
    system_prompt = build_system_prompt(latent_size)

    ds = load_dataset("json", data_files=json_path, split="train")

    # Create a stable 'images' list column containing absolute paths
    def add_images(ex):
        ex["images"] = [os.path.join(image_root, ex["image"])]
        return ex

    ds = ds.map(add_images)

    # Lazy decode list-of-paths -> list-of-PIL at batch time
    ds = ds.cast_column("images", Sequence(HFImage()))

    # Convert to GRPO dataset columns (and drop old columns)
    ds = ds.map(
        lambda ex: convert_example(ex, system_prompt), remove_columns=ds.column_names
    )

    # ----------------------------
    # 3) Train with GRPO
    # ----------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        train_dataset=ds,
        reward_funcs=reward_funcs,
        args=grpo_params,
    )

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser([GRPOParams, ImageDataset])
    grpo_params, image_dataset = parser.parse_args_into_dataclasses()
    main(
        grpo_params=grpo_params,
        json_path=image_dataset.json_path,
        image_root=image_dataset.image_root,
    )


# Things are I want do to:
# - Improve the rewards, like as of right now the accuracy could be heavily improved. We can also stop the generation after the </answer> also we can show the model in the system prompt the example of good answers, especially when requiring latex.
# - Possibly use a LLM judge to rate the answers instead of just accuracy.
# - Add more penalization rewards, like for example penalizing too long answers or answers that are not concise.
