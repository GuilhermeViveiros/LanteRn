import os

# Propagate SLURM distributed env vars before any torch/CUDA initialization.
# Only activate when actually running multi-process (SLURM_NTASKS > 1 or
# torchrun already set RANK/LOCAL_RANK). Without this, Accelerate sees every
# process as RANK=0 and HF Trainer falls back to DataParallel with a CPU model.
_slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 1))
_torchrun_active = "RANK" in os.environ or "LOCAL_RANK" in os.environ
if _slurm_ntasks > 1 or _torchrun_active:
    _local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    _rank       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  0)))
    _world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  1)))
    os.environ["LOCAL_RANK"] = str(_local_rank)
    os.environ["RANK"]       = str(_rank)
    os.environ["WORLD_SIZE"] = str(_world_size)
    # Restrict each process to its own GPU so HF Trainer never triggers DataParallel.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import logging
import torch
from transformers import HfArgumentParser
from termcolor import colored
from src.params import (TrainingParams, ModelParams, SFTDataParams)
from src.datasets.sft_data import make_sft_data_module, collate_fn_generate, SFTDataset
from src.datasets.sft_tetris_data import make_tetris_data_module, collate_fn_generate as tetris_collate_fn_generate, collate_fn_generate_ntp as tetris_collate_fn_generate_ntp
from src.datasets.family_sampler import FamilyGroupedDataset
from src.models import load_model
from src.trainer.sft_trainer import LantErnSFTrainer, ProgressBarLossLogger, VisCoTestLogger, LatentUtilityCallback, GenerationAccuracyCallback
from src.models.utils import get_last_checkpoint
from src.train import configure_vision_tower, configure_llm, configure_latent_only, set_latent_tokens
from src.utils import is_rank0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LantErn-Trainer")


def _save_data_diagnostics(dataset, output_dir: str, n_samples: int = 8):
    """
    Save a grid of sample images to output_dir/data_diagnostics/ before training starts.
    For each sample shows: [original image | corrupted main image | latent crop].
    Helps verify that corruption is applied correctly and latent crops use clean pixels.
    """
    import random
    from PIL import Image, ImageDraw, ImageFont
    from src.utils import center_and_crop_image

    diag_dir = os.path.join(output_dir, "data_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # Access the underlying SFTDataset through random_split wrapper if needed
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    indices = random.sample(range(len(base_dataset)), min(n_samples, len(base_dataset)))

    for i, idx in enumerate(indices):
        data = base_dataset.dataset[idx]
        if not data.get("bboxs"):
            continue

        img_orig = Image.open(data["img_path"]).convert("RGB")
        bbox = data["bboxs"][0]

        # Corrupted main image (same logic as __getitem__)
        img_corrupt = img_orig.copy()
        if base_dataset.corrupt_image:
            draw = ImageDraw.Draw(img_corrupt)
            for b in data["bboxs"]:
                x1, y1, x2, y2 = b
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        # Latent crop (always from original)
        crop = center_and_crop_image(img_orig, bbox)

        # Resize to fixed height for display
        h = 224
        def _resize_h(im):
            w = int(im.width * h / im.height)
            return im.resize((w, h))

        orig_r = _resize_h(img_orig)
        corrupt_r = _resize_h(img_corrupt)
        crop_r = _resize_h(crop)

        # Draw bbox on orig for reference
        orig_ann = orig_r.copy()
        scale_x = orig_r.width / img_orig.width
        scale_y = orig_r.height / img_orig.height
        x1, y1, x2, y2 = bbox
        ImageDraw.Draw(orig_ann).rectangle(
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            outline=(255, 0, 0), width=2
        )

        # Stitch side by side with labels
        pad = 4
        total_w = orig_ann.width + corrupt_r.width + crop_r.width + pad * 2
        grid = Image.new("RGB", (total_w, h + 16), (240, 240, 240))
        grid.paste(orig_ann, (0, 16))
        grid.paste(corrupt_r, (orig_ann.width + pad, 16))
        grid.paste(crop_r, (orig_ann.width + pad + corrupt_r.width + pad, 16))
        draw = ImageDraw.Draw(grid)
        draw.text((2, 2), "original (bbox in red)", fill=(0, 0, 0))
        draw.text((orig_ann.width + pad + 2, 2), "corrupted (model input)", fill=(0, 0, 0))
        draw.text((orig_ann.width + pad + corrupt_r.width + pad + 2, 2), "latent crop (clean)", fill=(0, 0, 0))

        grid.save(os.path.join(diag_dir, f"sample_{i:02d}.png"))

    logger.info(f"Data diagnostics saved to {diag_dir}/ ({len(indices)} samples)")


def train(training_params: TrainingParams, model_params: ModelParams, data_params: SFTDataParams):
    global local_rank
    if is_rank0():
        logger.info(f"Training model {model_params.model_id} with data from {data_params.data_path}")
        logger.info(colored(f"🚀 Training LantErn SFT model on {data_params.dataset_type} dataset", "green"))
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

    # check if we should resume training from a checkpoint
    resume_from_checkpoint = get_last_checkpoint(training_params.output_dir)
    if resume_from_checkpoint is not None:
        logger.info(colored(f"Resuming training from checkpoint: {resume_from_checkpoint}", "cyan"))
    else:
        logger.info(colored(f"Starting training from scratch", "cyan"))
        
    # set the latent tokens
    assert model_params.latent_size > 0 or model_params.latent_size == -1, "Latent size must be -1 for dynamic latent size or a positive integer"
    set_latent_tokens(processor, model, model_params.latent_size)
    

    # freeze specific components according to the training parameters
    if training_params.freeze_latent_only:
        configure_latent_only(model)
        trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
        logger.info(colored(f"Latent-only mode: {len(trainable)} trainable parameter tensors: {trainable}", "yellow"))
    else:
        configure_vision_tower(model, freeze_vision_tower=training_params.freeze_vision_tower, freeze_merger=training_params.freeze_merger)
        configure_llm(model, freeze_llm=training_params.freeze_llm)

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
    if data_params.dataset_type == "tetris":
        data_module = make_tetris_data_module(
            processor=processor,
            data_path=data_params.data_path,
            dummy=data_params.dummy,
            use_lvr=data_params.use_lvr,
            max_train_samples=data_params.max_train_samples,
            grayscale_intermediate=data_params.grayscale_intermediate,
        )
        if getattr(training_params, "use_family_batching", False):
            num_gpus = int(os.environ.get("WORLD_SIZE", 1))
            chunk_size = training_params.per_device_train_batch_size * num_gpus
            logger.info(f"[FamilyBatching] wrapping train dataset with FamilyGroupedIterableDataset "
                        f"(chunk_size={chunk_size}, key={training_params.family_batch_key})")
            data_module["train_dataset"] = FamilyGroupedDataset(
                dataset=data_module["train_dataset"],
                chunk_size=chunk_size,
                group_key=training_params.family_batch_key,
                seed=training_params.seed,
            )
        generate_collate = tetris_collate_fn_generate if data_params.use_lvr else tetris_collate_fn_generate_ntp
    elif data_params.dataset_type == "viscot":
        data_module = make_sft_data_module(
            processor=processor,
            data_path=data_params.data_path,
            dummy=data_params.dummy,
            split_percentages=data_params.split_percentages,
            corrupt_image=data_params.corrupt_image,
            corruption_type=data_params.corruption_type,
            filter_ids_path=data_params.filter_ids_path,
        )
        generate_collate = collate_fn_generate
        # Pre-training data diagnostics
        _save_data_diagnostics(data_module["train_dataset"], training_params.output_dir, n_samples=8)
    else: 
        raise ValueError(f"Invalid dataset type: {data_params.dataset_type}")

    
    callbacks = [ProgressBarLossLogger()]
    if data_params.dataset_type == "tetris" and "eval_dataset" in data_module:
        if data_params.use_lvr:
            callbacks.append(LatentUtilityCallback(
                eval_dataset=data_module["eval_dataset"],
                collate_fn=generate_collate,
                processor=processor,
                batch_size=training_params.per_device_eval_batch_size,
            ))
        else:
            callbacks.append(GenerationAccuracyCallback(
                eval_dataset=data_module["eval_dataset"],
                collate_fn=generate_collate,
                processor=processor,
                batch_size=training_params.per_device_eval_batch_size,
                max_eval_samples=300,
            ))
    if training_params.test_steps > 0 and "test_dataset" in data_module:
        callbacks.append(VisCoTestLogger(
            dataset=data_module.pop("test_dataset"),
            collate_fn=generate_collate,
            processor=processor, # necessary for the test script
            test_steps=training_params.test_steps,
            report_to="wandb"
        ))

    # Train
    trainer = LantErnSFTrainer(
        model=model,
        args=training_params,
        gamma=training_params.gamma,
        latent_only=training_params.freeze_latent_only,
        use_lvr=data_params.use_lvr,
        latent_loss_type=training_params.latent_loss_type,
        temperature=training_params.temperature,
        scheduled_sampling_prob=training_params.scheduled_sampling_prob,
        scheduled_sampling_warmup=training_params.scheduled_sampling_warmup,
        callbacks=callbacks,
        **data_module
    )

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    # save processor 
    processor.save_pretrained(training_params.output_dir)

    logger.info("Training completed")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingParams, ModelParams, SFTDataParams))
    training_params, model_params, data_params = parser.parse_args_into_dataclasses()
    train(training_params, model_params, data_params)