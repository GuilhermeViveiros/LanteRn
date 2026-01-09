import os
import logging
import torch
from transformers import HfArgumentParser
from termcolor import colored
from src.params import (TrainingParams, ModelParams, SFTDataParams)
from src.datasets.sft_data import make_sft_data_module, collate_fn_generate
from src.models import load_model
from src.trainer.sft_trainer import LantErnSFTrainer, ProgressBarLossLogger, VisCoTestLogger
from src.models.utils import get_last_checkpoint
from src.train import configure_vision_tower, configure_llm, set_latent_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")


def train(training_params: TrainingParams, model_params: ModelParams, data_params: SFTDataParams):
    global local_rank
    logger.info(f"Training model {model_params.model_id} with data from {data_params.data_path}")
    logger.info(colored(f"🚀 Training LantErn SFT model", "green"))
    logger.info(colored(f"Training parameters: {training_params}", "cyan"))
    logger.info(colored(f"🚀 Model parameters: {model_params}", "cyan"))
    logger.info(colored(f"Data parameters: {data_params}", "cyan"))


    compute_dtype = (torch.float16 if training_params.fp16 else (torch.bfloat16 if training_params.bf16 else torch.float32))
    logger.info(colored(f"Compute dtype: {compute_dtype}", "cyan"))

    # Load Model
    model, processor = load_model(
        model_params.model_id,
        compute_dtype=compute_dtype,
        use_cache=model_params.use_cache
    )
    print(f"model.config.vocab_size: {model.config.vocab_size}")

    # check if we should resume training from a checkpoint
    resume_from_checkpoint = get_last_checkpoint(training_params.output_dir)
    if resume_from_checkpoint is not None:
        logger.info(colored(f"Resuming training from checkpoint: {resume_from_checkpoint}", "cyan"))
    
    # set the latent tokens
    assert model_params.latent_size > 0 or model_params.latent_size == -1, "Latent size must be -1 for dynamic latent size or a positive integer"
    set_latent_tokens(processor, model, model_params.latent_size)
    

    # freeze specific components according to the training parameters
    configure_vision_tower(model, freeze_vision_tower=training_params.freeze_vision_tower, freeze_merger=training_params.freeze_merger)
    configure_llm(model, freeze_llm=training_params.freeze_llm)

    # Gradient Checkpointing
    if training_params.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # if eval_steps or test_steps are defined, ensure the split percentages are > 0
    if data_params.split_percentages[1] > 0:
        assert training_params.eval_steps > 0, "Eval steps must be greater than 0 if eval percentage is greater than 0 (data_params.split_percentages[1] > 0)"
    else:
        training_params.test_steps = 0
    if data_params.split_percentages[2] > 0:
        assert training_params.test_steps > 0, "Test steps must be greater than 0 if test percentage is greater than 0 (data_params.split_percentages[2] > 0)"
    else:
        training_params.test_steps = 0
    
    # Load data
    data_module = make_sft_data_module(
        processor=processor,
        data_path=data_params.data_path,
        dummy=data_params.dummy,
        split_percentages=data_params.split_percentages
    )

    
    callbacks = [ProgressBarLossLogger()]
    if training_params.test_steps > 0:
        callbacks.append(VisCoTestLogger(
            dataset=data_module.pop("test_dataset"), 
            collate_fn=collate_fn_generate,
            processor=processor, # necessary for the test script
            test_steps=training_params.test_steps,
            report_to="wandb"
        ))

    # Train
    trainer = LantErnSFTrainer(
        model=model,
        args=training_params,
        gamma=training_params.gamma,
        callbacks=callbacks,
        **data_module
    )

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    logger.info("Training completed")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingParams, ModelParams, SFTDataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)