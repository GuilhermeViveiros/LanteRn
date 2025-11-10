from dataclasses import dataclass, field

@dataclass
class ModelParams:
    model_name: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")

@dataclass
class TrainingParams:
    run_name: str = field(default="LantErn-SFT-Qwen2.5VL-3B")
    checkpoint_path: str = field(default="/mnt/data-artemis/gviveiros/lantern/checkpoints")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)

@dataclass
class DataParams:
    data_path: str = field(default="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json")