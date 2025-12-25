from trl import GRPOConfig
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class GRPOParams(GRPOConfig):
    # ------------------------------------------------------------------
    # Model (NONE in GRPOConfig)
    # ------------------------------------------------------------------
    model_path: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995")  # fmt: skip
    freeze_vision: bool = field(default=True)  # NOT in GRPOConfig
    freeze_vision_proj: bool = field(default=False)  # NOT in GRPOConfig

    # ------------------------------------------------------------------
    # Output / run identity
    # ------------------------------------------------------------------
    output_dir: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl")  # fmt: skip
    run_name: str = field(default="qwen25vl-grpo-lvr-night")
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

    # ------------------------------------------------------------------
    # Precision & compute
    # ------------------------------------------------------------------
    bf16: bool = field(default=True)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    learning_rate: float = field(default=5e-6)
    warmup_ratio: float = field(default=0.03)
    beta: float = field(default=0.1)

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    num_train_epochs: int = field(default=1)

    # ------------------------------------------------------------------
    # Generation / decoding
    # ------------------------------------------------------------------
    num_generations: int = field(default=4)
    max_completion_length: int = field(default=128)
    temperature: float = field(default=0.6)
    top_p: float = field(default=0.85)

    # ------------------------------------------------------------------
    # Low-level generation behavior
    # ------------------------------------------------------------------
    gradient_checkpointing: bool = field(default=False)
    use_cache: bool = field(default=True)  # NOT in GRPOConfig

    # ------------------------------------------------------------------
    # Logging / misc Trainer behavior
    # ------------------------------------------------------------------
    logging_steps: int = field(default=20)
    remove_unused_columns: bool = field(default=False)
    log_completions: bool = field(default=True)
    num_completions_to_print: Optional[int] = field(default=2)

    # ------------------------------------------------------------------
    # Rewards
    # - reward_names is NOT in GRPOConfig (trainer-side bookkeeping in your code)
    # - reward_weights: depending on TRL version, this may exist on GRPOConfig; if it doesn't, keeping it here is fine
    # ------------------------------------------------------------------
    reward_names: List[str] = field(default_factory=lambda: ["accuracy", "lvr_presence"])  # fmt: skip
    reward_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        gk = dict(getattr(self, "generation_kwargs", None) or {})
        gk.setdefault("use_cache", self.use_cache)
        self.generation_kwargs = gk


# @dataclass
# class GRPOParams:
#     # ------------------------------------------------------------------
#     # Model (non grpo config related)
#     # ------------------------------------------------------------------
#     model_path: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints/sft_mse_lt__lambda_0.1/checkpoint-995")  # fmt: skip
#     freeze_vision: bool = field(default=True)
#     freeze_vision_proj: bool = field(default=False)

#     # ------------------------------------------------------------------
#     # Output / run identity (model_path is here for convenience, not part of grpo config)
#     # ------------------------------------------------------------------
#     output_dir: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl")  # fmt: skip
#     run_name: str = field(default="qwen25vl-grpo-lvr-night")
#     report_to: Optional[List[str]] = field(default_factory=lambda: ["wandb"])

#     # ------------------------------------------------------------------
#     # Precision & compute
#     # ------------------------------------------------------------------
#     bf16: bool = field(default=True)

#     # ------------------------------------------------------------------
#     # Optimization
#     # ------------------------------------------------------------------
#     learning_rate: float = field(default=5e-6)
#     warmup_ratio: float = field(default=0.03)
#     beta: float = field(default=0.1)

#     # ------------------------------------------------------------------
#     # Batching
#     # ------------------------------------------------------------------
#     per_device_train_batch_size: int = field(default=2)
#     gradient_accumulation_steps: int = field(default=4)

#     # ------------------------------------------------------------------
#     # Training schedule
#     # ------------------------------------------------------------------
#     num_train_epochs: int = field(default=1)

#     # ------------------------------------------------------------------
#     # Generation / decoding
#     # ------------------------------------------------------------------
#     num_generations: int = field(default=4)
#     max_completion_length: int = field(default=128)
#     temperature: float = field(default=0.6)
#     top_p: float = field(default=0.85)

#     # ------------------------------------------------------------------
#     # Low-level generation behavior
#     # ------------------------------------------------------------------
#     use_cache: bool = field(default=True)

#     # ------------------------------------------------------------------
#     # Logging / misc Trainer behavior
#     # ------------------------------------------------------------------
#     logging_steps: int = field(default=5)
#     remove_unused_columns: bool = field(default=False)

#     # ------------------------------------------------------------------
#     # Rewards (Reward names are not used for GRPOConfig, but for the trainer)
#     # ------------------------------------------------------------------
#     reward_names: List[str] = field(default_factory=lambda: ["accuracy", "lvr_presence"])  # fmt: skip
#     reward_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

#     # ------------------------------------------------------------------
#     # Conversion
#     # ------------------------------------------------------------------
#     def to_grpo_config(self) -> GRPOConfig:
#         return GRPOConfig(
#             output_dir=self.output_dir,
#             run_name=self.run_name,
#             report_to=self.report_to,
#             bf16=self.bf16,
#             learning_rate=self.learning_rate,
#             warmup_ratio=self.warmup_ratio,
#             beta=self.beta,
#             per_device_train_batch_size=self.per_device_train_batch_size,
#             gradient_accumulation_steps=self.gradient_accumulation_steps,
#             num_train_epochs=self.num_train_epochs,
#             num_generations=self.num_generations,
#             max_completion_length=self.max_completion_length,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             logging_steps=self.logging_steps,
#             remove_unused_columns=self.remove_unused_columns,
#             reward_weights=self.reward_weights,
#             generation_kwargs={
#                 "use_cache": self.use_cache,
#             },
#         )


@dataclass
class ImageDataset:
    json_path: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/rl_dataset/lvr_data/virl39k.json")  # fmt: skip
    image_root: str = field(default="/mnt/scratch-hades/nunogoncalves/LantErn/rl_dataset/")  # fmt: skip
