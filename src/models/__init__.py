import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from src.models.qwen2_5VL import qwen2_5_mixed_modality_forward_lantern

def load_model(model_id, fp16: bool = True, bf16: bool = False):
    """
    Load the model and processor from the model id
    TODO: for now, we only support Qwen2.5-VL
    """

    if not "Qwen2.5-VL" in model_id:
        raise ValueError(f"Model {model_id} is not supported")
    
    if "Qwen2.5-VL-3B-Instruct" in model_id:
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lantern
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16
            if fp16 else (torch.bfloat16 if bf16 else None)
        )
        processor = AutoProcessor.from_pretrained(model_id)

        
    
    return model, processor