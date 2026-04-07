from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from transformers import TrainingArguments as HFTrainingArguments
from trl import GRPOConfig
from src.constants import (
    CHECKPOINTS_DIR, SCRATCH_NYX, SCRATCH_HADES,
    VISCOT_DATA_PATH, RL_DATA_PATH, RL_IMAGE_ROOT,
)

@dataclass
class ModelParams:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    latent_size: int = field(default=-1) # -1 for dynamic latent size (same as visual tokens)
    use_cache: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")
    
@dataclass
class TrainingParams(HFTrainingArguments):
    output_dir: str = field(default=CHECKPOINTS_DIR)
    num_train_epochs: int = field(default=1)
    save_steps: float = field(default=0.2)
    save_total_limit: int = field(default=2)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)
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
    freeze_latent_only: bool = field(default=False)  # if True, freeze everything except latent token embeddings
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    test_steps: int = field(default=100)
    dataloader_num_workers: int = field(default=4)
    dataloader_persistent_workers: bool = field(default=True)
    save_safetensors: bool = field(default=True)
    seed: int = field(default=42)
    #use_liger_kernel: bool = field(default=True) # LantErn does not support liger kernel
    #use_liger_kernel: bool = field(default=True)
    # per_device_train_batch_size: int = field(default=8)
    # gradient_accumulation_steps: int = field(default=1)

@dataclass
class GRPOArguments(GRPOConfig):
    # ------------------------------------------------------------------
    # Model Configuration
    # ------------------------------------------------------------------
    model_path: str = field(default=f"{SCRATCH_HADES}/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995")
    freeze_vision_tower: bool = field(default=True)
    freeze_merger: bool = field(default=True)
    freeze_llm: bool = field(default=False)

    # ------------------------------------------------------------------
    # Output / run identity
    # ------------------------------------------------------------------
    output_dir: str = field(default=f"{SCRATCH_NYX}/")
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    # ------------------------------------------------------------------
    # Precision & compute
    # ------------------------------------------------------------------
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    
    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    learning_rate: float = field(default=5e-6)
    warmup_ratio: float = field(default=0.03)
    beta: float = field(default=0.1)

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1) #steps_per_generation
    #steps_per_generation: int = field(default=1)

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    num_train_epochs: int = field(default=1)
    save_steps: int = field(default=300)
    max_steps: int = field(default=-1) # -1 for no max steps
    num_train_epochs: int = field(default=1)

    # ------------------------------------------------------------------
    # Generation / decoding
    # ------------------------------------------------------------------
    num_generations: int = field(default=2)
    max_completion_length: int = field(default=528)
    temperature: float = field(default=0.6)
    top_p: float = field(default=0.85)

    # ------------------------------------------------------------------
    # Low-level generation behavior
    # ------------------------------------------------------------------
    gradient_checkpointing: bool = field(default=False)
    # if True, cache is disabled

    # ------------------------------------------------------------------
    logging_steps: int = field(default=20)
    logging_strategy: str = field(default="steps")
    remove_unused_columns: bool = field(default=False)
    log_completions: bool = field(default=True)
    num_completions_to_print: Optional[int] = field(default=2)
    seed: int = field(default=42)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    reward_names: List[str] = field(default_factory=lambda: ["accuracy", "lvr_presence"])
    reward_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class SFTDataParams:
    data_path: str = field(default=VISCOT_DATA_PATH)
    dummy: bool = field(default=False)
    #split_percentages: Tuple[float, float, float] = field(default=(0.9, 0.1, 0.0))
    split_percentages: Tuple[float, float, float] = field(default=(0.9, 0.1, 0))
    corrupt_image: bool = field(default=False)
    corruption_type: str = field(default="bbox_blackout")
    filter_ids_path: Optional[str] = field(default=None)
    dataset_type: str = field(default="viscot",
                              metadata={"help": "viscot | tetris"})
    max_train_samples: Optional[int] = field(default=None,
                              metadata={"help": "Cap training set size. None = use all samples."})
    use_lvr: bool = field(default=True,
                          metadata={"help": "Use latent visual reasoning tokens (LantErn). "
                                            "Set False for NTP baseline."})

@dataclass
class RLDataParams:
    data_path: str = field(default=RL_DATA_PATH)
    image_root: str = field(default=RL_IMAGE_ROOT)