"""
Dataset for supervised fine-tuning (SFT)
"""
import os
from typing import Optional, Tuple
from functools import partial
import datasets
from datasets import load_dataset
import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, random_split
from typing import List
from src.utils import center_and_crop_image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
# import logger 
import logging
logger = logging.getLogger("LantErn-Dataset")

system_content = "You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. When you decide to generate abstract visual tokens, put them in <|lvr_start|>...<|lvr_end|>"

class SFTMonetDataset(Dataset):
    def __init__(
        self,
        processor: AutoProcessor,
        data_name: str = "NOVAglow646/Monet-SFT-125K",
        dummy: bool = False
    ):
        super(SFTMonetDataset, self).__init__()
        self.processor = processor
        

        # use datasets to load the dataset
        ds = load_dataset(data_name, split="train")
        print(f"Loaded dataset with {len(ds)} samples before filtering")
        # filter datasets where dataset_name != VisualCoT
        ds = ds.filter(lambda example: example["metadata"]["dataset_name"] == "Visual_CoT")
        print(f"Dataset has {len(ds)} samples after filtering out Visual_CoT")

        
        # TODO: fix this (harcoded for now)
        self.image_root = "/home/gviveiros/.cache/huggingface/hub/datasets--NOVAglow646--Monet-SFT-125K/snapshots/c77a2df9624e3e1e229dda90139453181e018eb8/"
        

        # define latent block
        self.latent_block = "<|lvr_start|><|lvr_sep|><|lvr_end|>"
        # if dummy, we only use the first 1000 examples
        if dummy:
            ds = ds.select(range(1000))

        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        metadata = sample.pop("metadata")
        data = sample.pop("data")
        dataset_name = metadata["dataset_name"]

       
        assistant_content = ""
        latent_visuals = []
        latent_blocks = 0
        
        for dt in data:
            # improve
            content, role = dt["content"], dt["role"]

        
            if role == "user":
                # Prefer clearer logic, explicit checks, and early exit for error diagnostics
                for obj in content:
                    if not isinstance(obj, dict):
                        continue
                    if "image" in obj and obj["image"]:
                        img_path = os.path.join(self.image_root, obj["image"])
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"User image path {img_path} does not exist")
                        img = Image.open(img_path)
                    if "text" in obj and obj["text"]:
                        question = obj["text"]

                if img is None or question is None:
                    raise ValueError(f"Missing required 'question' or 'image' in user message (question: {question}, image: {img})")
                
                # remove "\n" at the beginning from question if any
                if question.startswith("\n"):
                    question = question.lstrip("\n")

                user_content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ]
            elif role == "assistant":
                # get two objects from the content
                if len(content) < 2:
                    raise ValueError(f"Expected 2 or more objects for assistant content, got {len(content)}")
               
                for obj in content:
                    if obj["type"] == "text":
                        content = obj["text"]
                       
                        # replace all occurances of <abs_vis_token></abs_vis_token> by latent block
                        count = content.count("<abs_vis_token></abs_vis_token>")
                        if count > 0:
                            assistant_content += content.replace("<abs_vis_token></abs_vis_token>", self.latent_block)
                            latent_blocks += count
                        else:
                            assistant_content += content
                
                    elif obj["type"] == "image":
                        img_path = self.image_root + obj["image"]
                        if not os.path.exists(img_path):
                            raise ValueError(f"Image path {img_path} does not exist")
                        img = Image.open(img_path)
                        latent_visuals.append(img)
            else:
                continue

            assert latent_blocks == len(latent_visuals), f"Expected {latent_blocks} latent blocks, got {len(latent_visuals)}"
        
        
        return [
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
            {"role": "user","content": user_content},
            {"role": "assistant","content": [{"type": "text", "text": assistant_content}]},
            {"role": "assistant","content": [{"type": "image", "image": img} for img in latent_visuals]},
        ]

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    mask = (input_ids == image_token)
    return mask * 1

def collate_fn_generate(samples: List[dict], processor: AutoProcessor):
    # pop the last dict from the samples
    latent_visuals = [s.pop(-1) for s in samples]
    user_samples = [[s] for bs in samples for s in bs if s["role"] == "user"]
    labels = [s for bs in samples for s in bs if s["role"] == "assistant"]
    labels = [l["content"][0]["text"].split("<answer>")[-1].replace("</answer>","") for l in labels]
   
    image_inputs, video_inputs = process_vision_info(user_samples)
    text = processor.apply_chat_template(
        user_samples,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # ground truth latent images
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

    return (inputs, labels)

def collate_fn_sft(samples: List[dict], processor: AutoProcessor):
    # pop the last dict from the samples
    latent_visuals = [s.pop(-1) for s in samples]

    text = processor.apply_chat_template(samples, tokenize=False)
    image_inputs, video_inputs = process_vision_info(samples)
    
    nb_latent_visuals = sum(len(l["content"]) for l in latent_visuals)
    # replace the <|lvr_sep|> token with the number of latent tokens
    if nb_latent_visuals > 0:
        # process the latent images
        latent_text = processor.apply_chat_template(latent_visuals, tokenize=False)
        latent_image_inputs, _ = process_vision_info(latent_visuals)
        latent_inputs = processor(
            text=latent_text,
            images=latent_image_inputs,
            padding=True,
            return_tensors="pt",
        )
        latent_visuals = latent_inputs["pixel_values"]
        latent_grid_thw = latent_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size ** 2

        # Precompute num_latent_tokens efficiently using numpy where possible
        if processor.latent_size == -1:
            raise Exception("We dont support dynammic latent tokens yet")
            grid_prods = [int(g.prod()) if hasattr(g, "prod") else int(np.prod(g)) for g in latent_grid_thw]
            num_latent_tokens = [g // merge_length for g in grid_prods]
        else:
            num_latent_tokens = [processor.latent_size] * len(latent_grid_thw)
        
        # Replace <|lvr_sep|> efficiently
        lvr_sep = "<|lvr_sep|>"
        
        for idx, _ in enumerate(text):
            text[idx] = text[idx].replace(lvr_sep, lvr_sep * processor.latent_size, 1)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding="max_length",
        max_length=5000,
        truncation=True
    )

    #import pdb; pdb.set_trace()

    labels = torch.ones_like(inputs["input_ids"]) * -100
    # Pre-encode markers (fast, done once)
    # think and answer mut not be a special token
    assistant_start_tokens = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_end_tokens = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    


    def find_subsequence(seq: torch.Tensor, subseq: list) -> int:
        """
        Find the first start index of subseq inside a 1D tensor seq.
        Returns -1 if not found.
        """
        n, m = len(seq), len(subseq)
        if m == 0 or n < m:
            return -1

        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return -1

    # Process batch
    for i, ids in enumerate(inputs["input_ids"].tolist()):
        
        ts = find_subsequence(ids, assistant_start_tokens)
        te = ts + find_subsequence(ids[ts:], assistant_end_tokens)    
        assert ts != -1 and te != -1, "Markers missing in tokenization"

        start_pos = ts + len(assistant_start_tokens)
        end_pos = te + len(assistant_end_tokens) - 1 # remove the <|im_end|> token (1 token since its a special token)
        labels[i, start_pos:end_pos] = torch.tensor(ids[start_pos:end_pos], dtype=torch.long)
        #print(f"labels[i, start_pos:end_pos]: {processor.tokenizer.decode(labels[i, start_pos:end_pos], skip_special_tokens=False)}")

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.lvr_sep_id] = -100
    #print(f"labels != -100: {processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)}")
    inputs["labels"] = labels
    inputs["latent_mask_out"] = mask_image_output_tokens(inputs["input_ids"], processor.lvr_sep_id)
    # decode labels != -100
    # processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    # decoded_labels = processor.tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=False)
    # we are only interested in the latent images, so we return the latent inputs
    inputs["latent_values"] = latent_inputs["pixel_values"]
    inputs["latent_grid_thw"] = latent_inputs["image_grid_thw"]
    #import pdb; pdb.set_trace()
    return inputs
    
def make_sft_data_module(
    processor,
    dummy: bool = False,
    generate: bool = False,
    split_percentages: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    seed: int = 42,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTMonetDataset(processor=processor, dummy=dummy)

    # split the dataset into train, eval and test
    # test and val may be empty if the split percentages are 0
    assert sum(split_percentages) == 1, "Split percentages must sum to 1"
    assert split_percentages[0] > 0, "Train percentage must be greater than 0"
    
    train_percentage, \
        eval_percentage, \
            test_percentage = split_percentages

    train_size = int(train_percentage * len(sft_dataset))
    eval_size = int(eval_percentage * len(sft_dataset))
    test_size = int(test_percentage * len(sft_dataset))
    
    if test_size+eval_size+train_size < len(sft_dataset):
        eval_size += len(sft_dataset) - (test_size+eval_size+train_size) # add the remaining samples to the test_size

    logger.info(f"Total size: {len(sft_dataset)}, train: {train_size}, eval: {eval_size}, test: {test_size}")

    train_dataset, eval_dataset, test_dataset = random_split(sft_dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(seed))

    # set the collate function
    collate_fn = collate_fn_sft if not generate else collate_fn_generate

    out = {
        "train_dataset": train_dataset,
        "data_collator": partial(collate_fn, processor=processor)
    }   
    if eval_size > 0:
        out["eval_dataset"] = eval_dataset
    if test_size > 0:
        out["test_dataset"] = test_dataset
    # return the out dictionary
    return out


if __name__ == "__main__":
    from tqdm import tqdm
    # load the visual model
    from transformers import AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    # add special tokens for LantErn
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=False)
    processor.latent_size = 4
    # add token ids for new tokens
    processor.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    processor.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    processor.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")


    data_module = make_sft_data_module(processor=processor)

    # for sample in data_module["train_dataset"]:
    #     logger.info(sample)
    #     break
    
    # test with the dataloader
    from torch.utils.data import DataLoader
    batch_size = 1
    shuffle = True
    # Note: Removed the stray `per` from the argument list.
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator, 
        shuffle=shuffle, 
    )
    for i, batch in enumerate(tqdm(dataloader)):
        logger.info(f"Batch {i}: {batch.keys() if isinstance(batch, dict) else type(batch)}")
        import pdb; pdb.set_trace()
        if i == 0:
            break