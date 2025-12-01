from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelParams:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    latent_size: int = field(default=4)
    use_cache: bool = field(default=False)

@dataclass
class TrainingParams(HFTrainingArguments):
    run_name: str = field(default="LantErn-SFT-Qwen2.5VL-3B")
    output_dir: str = field(default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints")
    num_train_epochs: int = field(default=1)
    save_steps: int = field(default=100)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.1)
    # warmup_steps: int = field(default=500)
    gamma: float = field(default=0.1) # weight for the latpent similarity loss
    gradient_checkpointing: bool = field(default=True)
    fp16: bool = field(default=False)
    max_steps: int = field(default=-1) # -1 for no max steps
    bf16: bool = field(default=True)
    report_to: str = field(default="wandb")
    wandb_project: str = field(default="LantErn-SFT")
    wandb_entity: str = field(default="gviveiros")
    deepspeed: Optional[str] = field(default=None)
    freeze_vision_tower: bool = field(default=True)
    freeze_merger: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    test_steps: int = field(default=0)
    dataloader_num_workers: int = field(default=4)
    dataloader_persistent_workers: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    # per_device_train_batch_size: int = field(default=8)
    # gradient_accumulation_steps: int = field(default=1)




@dataclass
class DataParams:
    data_path: str = field(default="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json")
    dummy: bool = field(default=False)
    #split_percentages: Tuple[float, float, float] = field(default=(0.9, 0.0998, 0.0002))
    shuffle_dataset: bool = field(default=True)
    split_percentages: Tuple[float, float, float] = field(default=(0.9, 0.1, 0.0))