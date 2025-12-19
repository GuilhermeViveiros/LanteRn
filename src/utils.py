import time
from PIL import Image
from typing import List, Tuple
import torch
import torch.distributed as dist
import re

def is_rank0() -> bool:
    """Return True if current process is rank 0, or if not in distributed mode."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0 

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def center_and_crop_image(
    img: Image.Image,
    bbox: List[float],
    output_shape: Tuple[int, int] = None,
    context_scale: float = 1.2
) -> Image.Image:
    """
    Crop an image around a bounding box while preserving maximum resolution.

    Args:
        img: Original image (H, W, C).
        bbox: Bounding box [x1, y1, x2, y2].
        output_shape: Optional (height, width). If None, keep native cropped resolution.
        context_scale: Multiplier for adding context around bbox.

    Returns:
        Cropped (and optionally resized) image, and the transformation matrix.
    """
    H, W = img.height, img.width
    x1, y1, x2, y2 = bbox

    # Compute bbox center and size
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0

    # Apply context scaling
    w *= context_scale
    h *= context_scale

    # Compute new coordinates
    left = max(0, int(cx - w / 2.0))
    right = min(W, int(cx + w / 2.0))
    top = max(0, int(cy - h / 2.0))
    bottom = min(H, int(cy + h / 2.0))

    # Crop directly at native resolution
    cropped = img.crop((left, top, right, bottom))

    # Only resize if user explicitly wants an output shape
    if output_shape is not None:
        cropped = cropped.resize(output_shape)
    
    cropped.parent_filename = img.filename

    # save cropped image
    #cropped.save("img_bbox_0.jpg")
    return cropped


def extract_mc_answer(response: str) -> str:
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    
    if given_answer:
        match = re.search(r"(?:Answer:\s*)?(?:\(|\b)([A-Z])(?:\)|\b)", given_answer)
        if match:
            given_answer = match.group(1)
        else:
            given_answer = None
    
    return given_answer