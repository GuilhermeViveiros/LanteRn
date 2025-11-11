import logging
from transformers import HfArgumentParser
from src.params import (TrainingParams, ModelParams, DataParams)
from src.datasets.sft_data import make_sft_data_module

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src.trainer.trainer import LantErnSFTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")

def train(training_params: TrainingParams, model_params: ModelParams, data_params: DataParams):
    logger.info(f"Training model {model_params.model_id} with data from {data_params.data_path}")
    from termcolor import colored
    logger.info(colored(f"🚀 Training LantErn SFT model", "green"))
    # logger.info(colored(f"Training parameters: {training_params}", "cyan"))
    model_str = colored(f"🚀 Model parameters: {model_params}", "cyan")
    # logger.info(model_str)
    logger.info(colored(f"Data parameters: {data_params}", "cyan"))


    # Load Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_params.model_id)
    processor = AutoProcessor.from_pretrained(model_params.model_id)

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

    #resize the model
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Load data
    data_module = make_sft_data_module(
        vision_model=model.visual,
        processor=processor,
        data_path=data_params.data_path
    )

    # Train
    trainer = LantErnSFTrainer(
        model=model,
        args=training_params,
        **data_module
    )

    trainer.train()

    logger.info("Training completed")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingParams, ModelParams, DataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)