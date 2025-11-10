import logging
from transformers import HfArgumentParser
from src.params import (TrainingParams, ModelParams, DataParams)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Train")

def train(training_params: TrainingParams, model_params: ModelParams, data_params: DataParams):
    logger.info(f"Training model {model_params.model_name} with data from {data_params.data_path}")
    logger.info(f"Training parameters: {training_params}")
    logger.info(f"Model parameters: {model_params}")
    logger.info(f"Data parameters: {data_params}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_params.model_name)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingParams, ModelParams, DataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    # LantErn just supports Qwen models
    if "Qwen" not in model_params.model_name:
        raise ValueError(f"LantErn only supports Qwen models, got {model_params.model_name}")
    
    # Load data
    data = load_dataset("json", data_files=data_params.data_path)
    
    train(training_params, model_params, data_params)
    import pdb; pdb.set_trace()