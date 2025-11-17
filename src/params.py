from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelParams:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    latent_size: int = field(default=4)

@dataclass
class TrainingParams(HFTrainingArguments):
    run_name: str = field(default="LantErn-SFT-Qwen2.5VL-3B")
    output_dir: str = field(default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints")
    num_train_epochs: int = field(default=1)
    save_steps: int = field(default=5000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    gamma: float = field(default=0.1) # weight for the latent similarity loss
    gradient_checkpointing: bool = field(default=True)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    report_to: str = field(default="wandb")
    wandb_project: str = field(default="LantErn-SFT")
    wandb_entity: str = field(default="gviveiros")
    deepspeed: Optional[str] = field(default=None)



@dataclass
class DataParams:
    data_path: str = field(default="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json")
    dummy: bool = field(default=False)