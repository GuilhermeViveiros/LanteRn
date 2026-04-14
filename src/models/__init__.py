import os
import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src.models.qwen2_5VL.forward import qwen2_5_mixed_modality_forward_lantern
from src.lantern_generate.generate import generate as lantern_generate
import logging

logger = logging.getLogger("LantErn-Trainer")

def load_model(
    model_ref=None,
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

    assert model_ref, "Model ref must be defined, either a local path or HF path"

    use_local = False
    if os.path.isdir(str(model_ref)) or (os.path.exists(str(model_ref)) and os.path.isfile(os.path.join(str(model_ref), "config.json"))):
        # It's a local directory or contains config file
        use_local = True


    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lantern
    
    # If model_path is given, always load from the local folder (including local configs)
    # use flash attention 2 if use_liger_kernel is True
    #kwargs["attn_implementation"] = "flash_attention_2"
    kwargs["local_files_only"] = use_local
    logger.info(f"Loading model from {model_ref} with compute dtype {compute_dtype} and kwargs: {kwargs}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_ref,
        torch_dtype=compute_dtype,
        use_cache=use_cache,
        #attn_implementation="eager",
        #output_attentions=True,
        **kwargs
    )

    # replace this sample method with our own
    #processor = AutoProcessor.from_pretrained(model_ref, **kwargs)
    min_pixels = 256 * 28 * 28 # TODO: Add this variables in the params.py file
    max_pixels = 3500 * 28 * 28 # TODO: Add this variables in the params.py file

    try:
        processor = AutoProcessor.from_pretrained(
            model_ref, #"Qwen/Qwen2.5-VL-3B-Instruct",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs
        )
    except Exception as e:
        # Resolve to a local snapshot path to avoid network calls for the
        # mistral-regex check (which fires when pretrained_model_name_or_path
        # is a Hub ID rather than a local directory path).
        try:
            from huggingface_hub import snapshot_download
            local_qwen_path = snapshot_download(
                "Qwen/Qwen2.5-VL-3B-Instruct", local_files_only=True
            )
        except Exception:
            local_qwen_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        proc_kwargs = {k: v for k, v in kwargs.items() if k != "local_files_only"}
        processor = AutoProcessor.from_pretrained(
            local_qwen_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True,
            **proc_kwargs
        )
    #else:
    #    raise ValueError(f"Only Qwen2.5-VL-3B-Instruct is currently supported (got {model_ref})")

    return model, processor