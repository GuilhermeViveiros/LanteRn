import logging
from transformers import HfArgumentParser
from src.params import (TrainingParams, ModelParams, DataParams)
from src.datasets.sft_data import make_sft_data_module

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import SFTConfig
from src.trainer.trainer import LantErnSFTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Train")

def train(training_params: TrainingParams, model_params: ModelParams, data_params: DataParams):
    logger.info(f"Training model {model_params.model_name} with data from {data_params.data_path}")
    logger.info(f"Training parameters: {training_params}")
    logger.info(f"Model parameters: {model_params}")
    logger.info(f"Data parameters: {data_params}")


    # Load Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_params.model_id)
    processor = AutoProcessor.from_pretrained(model_params.model_id)
    
    # Load data
    train_dataset, \
        eval_dataset, \
            data_collator = make_sft_data_module(
                vision_model=model.vision_model,
                processor=processor,
                data_path=data_params.data_path
            )

    # Train
    trainer = LantErnSFTrainer(
        model,
        args=SFTConfig(max_length=None),
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        training_args=training_params
    )

    trainer.train()
