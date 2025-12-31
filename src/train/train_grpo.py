import os
import logging
import torch
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from termcolor import colored
from src.params import (GRPOArguments, ModelParams, RLDataParams)
from src.datasets.grpo_data import GRPODataset
from src.models import load_model
from src.train import configure_vision_tower, configure_llm
from src.trainer.grpo_trainer import LantErnGRPOTrainer
from src.models.utils import get_last_checkpoint
#from src.reward_funcs import format_reward
#from src.rl.rewards import build_accuracy_reward, build_structure_reward
from src.train import set_latent_tokens

# custom rl utils
from src.rl.utils import convert_example
from src.rl.prompt import build_system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")

from src.rl import rewards as rewards_module


def build_reward_funcs(grpo_params: GRPOArguments, model):  # noqa: F811
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



def train(grpo_params: GRPOArguments, model_params: ModelParams, data_params: RLDataParams):
    global local_rank
    logger.info(f"Training model {model_params.model_id} with data from {data_params.data_path}")
    logger.info(colored(f"🚀 Training LantErn RL stage: GRPO ", "green"))
    #logger.info(colored(f"Training parameters: {grpo_params}", "cyan"))
    logger.info(colored(f"🚀 Model parameters: {model_params}", "cyan"))
    logger.info(colored(f"Data parameters: {data_params}", "cyan"))

    # compute type
    compute_dtype = (torch.float16 if grpo_params.fp16 else (torch.bfloat16 if grpo_params.bf16 else torch.float32))
    logger.info(colored(f"Compute dtype: {compute_dtype}", "cyan"))

    # load model
    model, processor = load_model(
        grpo_params.model_path,
        compute_dtype=compute_dtype,
        use_cache=model_params.use_cache
    )

    ref_model, _ = load_model(
        grpo_params.model_path,
        compute_dtype=compute_dtype,
        use_cache=model_params.use_cache
    )
    ref_model.requires_grad_(False)
    ref_model.eval()

    
    # set the latent tokens
    assert model.config.latent_size > 0 or model.config.latent_size == -1, "Latent size must be -1 for dynamic latent size or a positive integer"
    set_latent_tokens(processor, model, model.config.latent_size, special_tokens=False)
    set_latent_tokens(processor, ref_model, model.config.latent_size, special_tokens=False)
    
    
    # freeze specific components according to the training parameters
    configure_vision_tower(model, freeze_vision_tower=grpo_params.freeze_vision_tower, freeze_merger=grpo_params.freeze_merger)
    configure_llm(model, freeze_llm=grpo_params.freeze_llm)

    # Gradient Checkpointing
    if grpo_params.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    latent_size = int(model.config.latent_size)
    logger.info(colored(f"Latent size: {latent_size}", "cyan"))
    system_prompt = build_system_prompt(latent_size)
    train_dataset = GRPODataset(
        data_path=data_params.data_path,
        image_root=data_params.image_root,
        system_prompt=None
    )

    # prepare rewards
    reward_funcs = build_reward_funcs(grpo_params, model)

    
    # Train
    trainer = LantErnGRPOTrainer(
        model=model,
        ref_model=ref_model,
        latent_size=latent_size,
        args=grpo_params,
        processing_class=processor,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs
    )

    trainer.train()

    logger.info("Training completed")

if __name__ == "__main__":
    parser = HfArgumentParser((GRPOArguments, ModelParams, RLDataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)