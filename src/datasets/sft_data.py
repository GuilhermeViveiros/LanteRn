"""
Dataset for supervised fine-tuning (SFT)
"""
from functools import partial
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Union
import torch
from src.utils import center_and_crop_image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        vision_model: torch.nn.Module,
        processor: AutoProcessor
    ):
        super(SFTDataset, self).__init__()
        self.processor = processor
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # retrieve the image
        data = self.dataset[idx]
        
        # Extract data fields
        question = data["question"]
        reasoning_traces = data["reasoning_traces"]
        answer = reasoning_traces["answer"]
        pre_visual_latent_reasoning = reasoning_traces.get("pre_visual_text_think", None)
        post_visual_latent_reasoning = reasoning_traces.get("post_visual_text_think", None)
        text_only_reasoning = reasoning_traces.get("text_think", None)
        
        # Validate reasoning traces
        if text_only_reasoning is None:
            assert post_visual_latent_reasoning is not None or pre_visual_latent_reasoning is not None, \
                "If text_reasoning is not None, post_visual_latent_reasoning or pre_visual_latent_reasoning must be not None"
            
        # Extract the image and process bboxes
        img = Image.open(data["img_path"])
        # Build user content
        user_content = [
            {"type": "text", "text": question},
            {"type": "image", "image": img},
        ]

        # Build assistant content
        assistant_content = []
        if text_only_reasoning is not None:
            # just text without latent visual reasoning
            assistant_content.append({"type": "text", "text": text_only_reasoning})
        else:

            img_bboxs = [center_and_crop_image(img, bbox) for bbox in data["bboxs"]]
            # Validate bbox count matches post_reasoning_trace count
            assert len(img_bboxs) == len(post_visual_latent_reasoning), \
                f"The number of bboxs ({len(img_bboxs)}) and post_visual_latent_reasoning ({len(post_visual_latent_reasoning)}) must be the same"


            # Build assistant content by adding pre_visual_latent_reasoning, and interleaving bbox images with post_visual_latent_reasoning
            if pre_visual_latent_reasoning is not None:
                assistant_content.append({"type": "text", "text": pre_visual_latent_reasoning})
            for img_bbox, post_visual_latent_reasoning in zip(img_bboxs, post_visual_latent_reasoning, strict=True):
                assistant_content.append({"type": "image", "image": img_bbox})
                assistant_content.append({"type": "text", "text": post_visual_latent_reasoning})
            
        # Add the final answer
        assistant_content.append({"type": "text", "text": answer})

        # Create the message
        message = [{
            "role": "user",
            "content": user_content,
        }, {
            "role": "assistant",
            "content": assistant_content,
        }]

        return message

def collate_fn(samples: List[str], processor: AutoProcessor):
    # apply the tokenizer
    text = processor.apply_chat_template(samples, tokenize=False)
    # process the vision info
    image_inputs, video_inputs = process_vision_info(samples)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    import pdb; pdb.set_trace()
    return inputs
    
def make_sft_data_module(vision_model, processor, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDataset(
        data_path=data_path, processor=processor, vision_model=vision_model
    )

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": None,
        "data_collator": partial(collate_fn, processor=processor)
    }   

if __name__ == "__main__":
    
    data_path="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"
    

    # load the visual model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #    model_id, torch_dtype="auto", device_map="auto"
    #)
    processor = AutoProcessor.from_pretrained(model_id)
    data_module = make_sft_data_module(
        vision_model=None, 
        processor=processor, 
        data_path=data_path
    )

    # test with the dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(data_module["train_dataset"], batch_size=8, collate_fn=data_module["data_collator"])
    for batch in dataloader:
        print(batch.keys())
        break
    
    import pdb; pdb.set_trace()