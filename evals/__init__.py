import torch
from typing import List
from transformers import AutoProcessor
from functools import partial
from src.lantern_generate.generate import generate as lantern_generate
from src.models.utils import apply_latent_compression
from src.decorators import measure_time

# Run batch Inference
@torch.no_grad()
@measure_time
def run_batch_inference(
    model,
    inputs,
    use_lvr: bool = True,
    use_gt: bool = False
):

    # get gt latent values
    if "latent_values" in inputs:
        gt_latent_embeds = apply_latent_compression(
            model,
            input_ids=inputs["input_ids"],
            latent_values=inputs.pop("latent_values") if "latent_values" in inputs else None,
            latent_grid_thw=inputs.pop("latent_grid_thw") if "latent_grid_thw" in inputs else None,
            latent_size=model.config.latent_size,
        )

    inputs = inputs.to(model.device)
        
    # I'll pass the ground truth latent embeddings to the generate function for debugging purposes
    # this will be removed in the future (just for stress testing purposes)
    return model.generate(
        **inputs,
        max_new_tokens=526,
        do_sample=False,
        custom_generate=partial(
            lantern_generate,
            gt_latent_embeds=gt_latent_embeds if use_gt else None
        ) if use_lvr else None,
        use_cache=True,
    )
    