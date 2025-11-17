"""
Dataset for supervised fine-tuning (SFT)
"""
from functools import partial
import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List
from src.utils import center_and_crop_image, measure_time
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        dummy: bool = False,
    ):
        super(SFTDataset, self).__init__()
        self.processor = processor
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        # if dummy, we only use the first 1000 examples
        if dummy:
            import random
            self.dataset = random.sample(self.dataset, min(100, len(self.dataset)))
            # self.dataset = self.dataset[:1000]
    
        def pre_validation(data):
            # ignore samples with more than 1 bbox
            if len(data["bboxs"]) > 1:
                return False
            return True
        self.dataset = [data for data in self.dataset if pre_validation(data)]

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
        latent_visuals = []
        if text_only_reasoning is not None:
            # just text without latent visual reasoning
            assistant_content = "<think>" + text_only_reasoning + "</think>"
        else:
            assistant_content = "<think>"
            img_bboxs = [center_and_crop_image(img, bbox) for bbox in data["bboxs"]]
            # Validate bbox count matches post_reasoning_trace count
            assert len(img_bboxs) == len(post_visual_latent_reasoning), \
                f"The number of bboxs ({len(img_bboxs)}) and post_visual_latent_reasoning ({len(post_visual_latent_reasoning)}) must be the same for example {idx}"


            # Build assistant content by adding pre_visual_latent_reasoning, and interleaving bbox images with post_visual_latent_reasoning
            if pre_visual_latent_reasoning is not None:
                assistant_content += pre_visual_latent_reasoning
            for img_bbox, post_visual_latent_reasoning in zip(img_bboxs, post_visual_latent_reasoning, strict=True):
                latent_visuals.append(img_bbox)
                # TODO: change different choices of latent tokens in the future
                # lets inject the custom tokens - at this always inject 4 latent tokens
                assistant_content += f"<|lvr_start|><|lvr_sep|><|lvr_sep|><|lvr_sep|><|lvr_sep|><|lvr_end|>"
                assistant_content += post_visual_latent_reasoning
            
        # Add the final answer
        assistant_content += "</think>" + "<answer>" + answer + "</answer>"

        # user, assistant, and latent images
        return [
            {"role": "user","content": user_content},
            {"role": "assistant","content": [{"type": "text", "text": assistant_content}]},
            {"role": "assistant","content": [{"type": "image", "image": img} for img in latent_visuals]} # latent images to be removed afer
        ]

def collate_fn(samples: List[dict], processor: AutoProcessor):
    # pop the last dict from the samples
    latent_visuals = [s.pop(-1) for s in samples]
    text = processor.apply_chat_template(samples, tokenize=False)

    image_inputs, video_inputs = process_vision_info(samples)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    labels = torch.ones_like(inputs["input_ids"]) * -100
    for i, t in enumerate(text):
        assistant_start = t.find("<|im_start|>assistant")
        if assistant_start >= 0:
            assistant_part = t[assistant_start:]
            assistant_ids = processor.tokenizer(assistant_part).input_ids
            labels[i, -len(assistant_ids):] = torch.tensor(assistant_ids, dtype=torch.long)
    
    labels[labels == processor.tokenizer.pad_token_id] = -100
    lvr_sep_id = processor.tokenizer.encode("<|lvr_sep|>")[0]
    labels[labels == lvr_sep_id] = -100
    inputs["labels"] = labels
    
    nb_latent_visuals = sum(len(l["content"]) for l in latent_visuals)
    if nb_latent_visuals > 0:
        # process the latent images
        latent_text = processor.apply_chat_template(latent_visuals, tokenize=False)
        latent_image_inputs, latent_video_inputs = process_vision_info(latent_visuals)
        latent_inputs = processor(
            text=latent_text,
            images=latent_image_inputs,
            videos=latent_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # we are only interested in the latent images, so we return the latent inputs
        inputs["latent_values"] = latent_inputs["pixel_values"]
        inputs["latent_grid_thw"] = latent_inputs["image_grid_thw"]
    
    
    return inputs
    
def make_sft_data_module(processor, data_path, dummy: bool = False, **kwargs):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDataset(
        data_path=data_path, processor=processor, dummy=dummy
    )

    return {
        "train_dataset": sft_dataset,
        "eval_dataset": None,
        "data_collator": partial(collate_fn, processor=processor)
    }   

if __name__ == "__main__":
    
    data_path="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"
    

    # load the visual model
    from transformers import AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    # add special tokens for LantErn
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

    data_module = make_sft_data_module(
        processor=processor, 
        data_path=data_path
    )

    # test with the dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(data_module["train_dataset"], batch_size=6, collate_fn=data_module["data_collator"])
    for batch in dataloader:
        print(batch.keys())
        break
    