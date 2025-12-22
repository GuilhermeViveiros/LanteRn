import os
import logging
import torch
from transformers import HfArgumentParser
from termcolor import colored
from src.params import (TrainingParams, ModelParams, DataParams)
from src.datasets.sft_data import make_sft_data_module, collate_fn_generate
from src.models import load_model
from src.trainer.sft_trainer import LantErnSFTrainer, ProgressBarLossLogger, VisCoTestLogger
from src.models.utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def set_latent_tokens(processor, model, latent_size: int, special_tokens: bool = True):
    # add special tokens for LantErn
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=special_tokens)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=special_tokens)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=special_tokens)
    # set the latent size
    processor.latent_size = latent_size
    model.config.latent_size = latent_size

    model.config.additional_special_tokens = [
        "<|lvr_start|>",
        "<|lvr_sep|>",
        "<|lvr_end|>"
    ]

    # resize the model embeddings size
    model.resize_token_embeddings(len(processor.tokenizer))

    # get the ids of the special tokens -> used for the model and processor
    lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    processor.lvr_start_id, processor.lvr_sep_id, processor.lvr_end_id = lvr_start_id, lvr_sep_id, lvr_end_id
    model.config.lvr_start_id, model.config.lvr_sep_id, model.config.lvr_end_id = lvr_start_id, lvr_sep_id, lvr_end_id
    
    
def configure_vision_tower(model):
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_params.freeze_vision_tower)
    logger.info(colored(f"Freezing vision tower: {training_params.freeze_vision_tower}", "cyan"))
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_params.freeze_merger)
    logger.info(colored(f"Freezing merger: {training_params.freeze_merger}", "cyan"))

def configure_llm(model):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_params.freeze_llm)
    logger.info(colored(f"Freezing LLM Head: {training_params.freeze_llm}", "cyan"))

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_params.freeze_llm)
    logger.info(colored(f"Freezing LLM: {training_params.freeze_llm}", "cyan"))


def train(training_params: TrainingParams, model_params: ModelParams, data_params: DataParams):
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
    configure_vision_tower(model)
    configure_llm(model)

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
    parser = HfArgumentParser((TrainingParams, ModelParams, DataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)