import logging
import torch
from transformers import HfArgumentParser
from src.params import (TrainingParams, ModelParams, DataParams)
from src.datasets.sft_data import make_sft_data_module
from src.models import load_model
from src.trainer.sft_trainer import LantErnSFTrainer, ProgressBarLossLogger, EvalLossLogger, VisCoTEvalLogger
from termcolor import colored


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

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
    model_str = colored(f"🚀 Model parameters: {model_params}", "cyan")
    logger.info(colored(f"Data parameters: {data_params}", "cyan"))


    compute_dtype = (torch.float16 if training_params.fp16 else (torch.bfloat16 if training_params.bf16 else torch.float32))
    logger.info(colored(f"Compute dtype: {compute_dtype}", "cyan"))

    # Load Model
    model, processor = load_model(model_params.model_id, compute_dtype=compute_dtype, use_cache=model_params.use_cache)

    # add special tokens for LantErn
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

    model.config.additional_special_tokens = [
        "<|lvr_start|>",
        "<|lvr_sep|>",
        "<|lvr_end|>"
    ]

    model.config.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    model.config.latent_size = model_params.latent_size

    #resize the model
    model.resize_token_embeddings(len(processor.tokenizer))
    
    configure_vision_tower(model)
    configure_llm(model)

        
    # Gradient Checkpointing
    if training_params.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load data
    data_module = make_sft_data_module(
        processor=processor,
        data_path=data_params.data_path,
        dummy=data_params.dummy,
    )
    # pop the test dataset
    test_dataset = data_module.pop("test_dataset")

    # check if wandb is enabled
    if training_params.report_to == "wandb":
        logger.info(colored("Initializing WandB...", "yellow"))
        import wandb
        wandb.init(
            project=training_params.wandb_project, 
            #name=training_params.run_name,
            config=training_params
        )
    callbacks = [ProgressBarLossLogger(), EvalLossLogger()]
    if test_dataset is not None:
        callbacks.append(VisCoTEvalLogger(test_dataset))

    # Train
    trainer = LantErnSFTrainer(
        model=model,
        args=training_params,
        callbacks=callbacks,
        **data_module
    )

    trainer.train()

    logger.info("Training completed")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingParams, ModelParams, DataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)