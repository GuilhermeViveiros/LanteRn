import random
import torch
from typing import List
from PIL import Image
from transformers import AutoProcessor
from functools import partial
from qwen_vl_utils import process_vision_info
from src.lantern_generate.generate import generate as lantern_generate
from src.models.utils import apply_latent_compression
from src.decorators import measure_time


def random_crop_bbox(img: Image.Image, crop_ratio: float = 0.4):
    """Generate a random bbox covering ~crop_ratio of the image dimensions."""
    W, H = img.width, img.height
    bw, bh = int(W * crop_ratio), int(H * crop_ratio)
    x1 = random.randint(0, max(0, W - bw))
    y1 = random.randint(0, max(0, H - bh))
    return [x1, y1, x1 + bw, y1 + bh]


def get_gt_latent_values(cropped_images, processor):
    """Encode a list of cropped PIL images into pixel_values + image_grid_thw."""
    messages = [
        [{
            "role": "assistant",
            "content": [{"type": "image", "image": img[0]}]
        }] for img in cropped_images
    ]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs["pixel_values"], inputs["image_grid_thw"]

# Run batch Inference
@torch.no_grad()
@measure_time
def run_batch_inference(
    model,
    inputs,
    return_dict: bool = True,
    output_attentions: bool = True,
    use_lvr: bool = True,
    use_gt: bool = False,
):

    # get gt latent values
    if "latent_values" in inputs and use_gt:
        # get ground truth embeddings
        gt_latent_embeds = apply_latent_compression(
            model,
            latent_values=inputs.pop("latent_values") if "latent_values" in inputs else None,
            latent_grid_thw=inputs.pop("latent_grid_thw") if "latent_grid_thw" in inputs else None,
            latent_size=model.config.latent_size,
        )

    if "latent_values" in inputs:
        raise Exception("not allowed, inference, debug")

    inputs = inputs.to(model.device)
    
    assert "latent_values" not in inputs, "remove latent values from inputs during inference"

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
        output_attentions=output_attentions,
        return_dict_in_generate=return_dict
    )
    