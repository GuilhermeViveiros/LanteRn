import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src.models.qwen2_5VL.forward import qwen2_5_mixed_modality_forward_lantern
from src.lantern_generate.generate import generate as lantern_generate
import logging

logger = logging.getLogger("LantErn-Trainer")

def load_model(
    model_id=None,
    model_path=None,
    compute_dtype: torch.dtype = torch.float16,
    use_cache: bool = False,
    **kwargs
):
    """
    Load the model and processor from a model id (remote hub) or local model path.
    If model_path is provided, load locally from there (including any configs).
    Otherwise, fall back to using model_id.
    TODO: Check if this model is supported (currently only Qwen2.5-VL-*B-Instruct is supported)
    """

    model_ref = model_path if model_path is not None else model_id
    if model_ref is None:
        raise ValueError("Must provide either model_id or model_path.")

    #if not "Qwen2.5-VL" in str(model_ref):
    #    raise ValueError(f"Model {model_ref} is not supported")

    #if "Qwen2.5-VL-3B-Instruct" in str(model_ref):
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lantern
    
    # If model_path is given, always load from the local folder (including local configs)
    # use flash attention 2 if use_liger_kernel is True
    #kwargs["attn_implementation"] = "flash_attention_2"
    logger.info(f"Loading model from {model_ref} with compute dtype {compute_dtype} and kwargs: {kwargs}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_ref,
        torch_dtype=compute_dtype,
        use_cache=use_cache,
        **kwargs
    )

    # replace this sample method with our own
    #processor = AutoProcessor.from_pretrained(model_ref, **kwargs)
    min_pixels = 256 * 28 * 28 # TODO: Add this variables in the params.py file
    max_pixels = 3500 * 28 * 28 # TODO: Add this variables in the params.py file


    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        **kwargs
    )
    #else:
    #    raise ValueError(f"Only Qwen2.5-VL-3B-Instruct is currently supported (got {model_ref})")

    return model, processor