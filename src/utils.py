from PIL import Image
from typing import List, Tuple
import time

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
    # save cropped image
    #cropped.save("img_bbox_0.jpg")
    return cropped


# decorator that measures the time of the function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper