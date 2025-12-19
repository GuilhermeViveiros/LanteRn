"""
Dataset for supervised fine-tuning (SFT)
"""
from typing import Optional, Tuple
from functools import partial
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List
from src.utils import center_and_crop_image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# import logger 
import logging
logger = logging.getLogger("LantErn-Dataset")

class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        dummy: bool = False,
        use_lvr: bool = True,
    ):
        super(SFTDataset, self).__init__()
        self.processor = processor
        self.use_lvr = use_lvr
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        # remove sample textvqa/34084d4c3c347b83.jpg
        for data in self.dataset: # MINOR BUGG: ignore this sample for now
            if data["img_path"] == "/mnt/data-artemis/gviveiros/lantern/textvqa/34084d4c3c347b83.jpg":
                self.dataset.remove(data)

        def pre_validation(data, idx):
            # ignore samples with more than 1 bbox
            if len(data["bboxs"]) > 1:
                return False
            return True
    
        
        logger.info(f"Number of examples of VisCoT data: {len(self.dataset)}")
        # remove cases where the image is too large and bboxs are more than 1
        self.dataset = [data for data, idx in zip(self.dataset, range(len(self.dataset))) if pre_validation(data, idx)]
        logger.info(f"Number of examples of VisCoT data after removing examples with more than 1 bbox: {len(self.dataset)}")
        #self.dataset = [data for data, idx in zip(self.dataset, range(len(self.dataset))) if filter_too_large_images(data, idx)]
        #logger.info(f"Number of examples of VisCoT data after removing examples with too large images: {len(self.dataset)}")

        # if dummy, we only use the first 1000 examples
        if dummy:
            #import random
            #self.dataset = random.sample(self.dataset, min(5000, len(self.dataset)))
            self.dataset = self.dataset[:1000]
        
        #self.dataset = self.dataset[:100]

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
    
        # here maybe add the bbox coordinates or the image if the variable prefix is not None
        # cropped_imgs = [center_and_crop_image(img, bbox) for bbox in data["bboxs"]]
        # version 1 uses only bboxs
        #question += "".join(f"\nYou can answer the question by looking at the region {data['bboxs'][0]} in the image.")
    
        # version 2 uses the bboxs and the image
        #cropped_img = center_and_crop_image(img, data["bboxs"][0])
        # user_content = [
        #     {"type": "text", "text": question},
        #     {"type": "image", "image": cropped_img},
        #     #{"type": "text", "text": "You can answer the question by looking at the following image region:"},
        #     #{"type": "image", "image": cropped_img},
        # ]

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
                assistant_content += "<|lvr_start|>"+'<|lvr_sep|>'*self.latent_size + "<|lvr_end|>"
                assistant_content += post_visual_latent_reasoning
            
        # Add the final answer
        assistant_content += "</think>" + "<answer>" + answer + "</answer>"
        
        return [
            {"role": "user","content": user_content},
            {"role": "assistant","content": [{"type": "text", "text": assistant_content}]},
            {"role": "assistant","content": [{"type": "image", "image": img} for img in latent_visuals]}, # latent images to be removed afer
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
        #inputs["labels"] = labels

    return (inputs, labels)

def collate_fn_sft(samples: List[dict], processor: AutoProcessor):
    # pop the last dict from the samples
    latent_visuals = [s.pop(-1) for s in samples]
    text = processor.apply_chat_template(samples, tokenize=False)

    image_inputs, video_inputs = process_vision_info(samples)
    
    #print([len(t) for t in text])
    #print(image_inputs)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding="max_length",
        max_length=5000,
        truncation=True
    )

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
    lvr_sep_id = processor.tokenizer.encode("<|lvr_sep|>")[0]
    labels[labels == lvr_sep_id] = -100
    inputs["labels"] = labels
    inputs["latent_mask_out"] = mask_image_output_tokens(inputs["input_ids"], lvr_sep_id)
    

    # print("inputs: ", processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    
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
    
def make_sft_data_module(
    processor,
    data_path: str,
    latent_size: int,
    dummy: bool = False,
    generate: bool = False,
    split_percentages: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    seed: int = 42,
    per_device_eval_batch_size: int = 1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDataset(
        data_path=data_path, processor=processor, dummy=dummy, latent_size=latent_size
    )

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

    train_dataset, eval_dataset, _ = random_split(sft_dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(seed))

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
    
    # HARDOCDED FOR NOW: we'll repace the testdataset with a custom version with MC questions
    #from evals.eval import MCDataset, collate_fn_mc
    #mc_dataset = MCDataset(["viscot", "vstar", "blink"])
    #dataloader = DataLoader(data, batch_size=per_device_eval_batch_size, shuffle=True, collate_fn=partial(collate_fn_mc, processor=processor))
    #logger.info(f"Number of samples in custom test dataloader: {len(dataloader)}")
    # out["test_dataset"] = mc_dataset.data
    return out

if __name__ == "__main__":
    from tqdm import tqdm
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

    for sample in data_module["train_dataset"]:
        logger.info(sample)
        break

    # test with the dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(data_module["train_dataset"], batch_size=per_device_eval_batch_size, collate_fn=data_module["data_collator"], shuffle=False)
    sizes = []
    for batch in tqdm(dataloader):
        sizes.append(batch["input_ids"].shape[1])
    
    # log some stats about the sizes, the quantiles, the mean, the std, the max, the min
    import numpy as np
    logger.info(f"Sizes: {sizes}")
    logger.info(f"Quantiles: {np.quantile(sizes, [0.25, 0.5, 0.75])}")
    logger.info(f"Mean: {np.mean(sizes)}")
    logger.info(f"Std: {np.std(sizes)}")
    logger.info(f"Max: {np.max(sizes)}")
    logger.info(f"Min: {np.min(sizes)}")
