"""
SFT dataset for the Tetris visual analogy task.

Schema (one record per sample):
  {
    "sample_id": int,
    "img_path": "/path/to/composite.png",       # A:B::C:? image
    "answer": "a"|"b"|"c"|"d",
    "bboxs": [[x1,y1,x2,y2], ...],              # option panel coords (eval only)
    "reasoning_traces": {
      "pre_visual_text_think": "...",            # text before the latent token
      "intermediate_img_path": "/path/to/strip.png",  # rotation strip = latent visual
      "post_visual_text_think": ["","text","",""],     # 4 entries, only correct is filled
      "text_think": "...",                       # text after latent block
      "answer": "a"|"b"|"c"|"d"
    }
  }

The assistant turn is assembled as:
  <think>
    {pre_visual_text_think}
    <|lvr_start|><|lvr_sep|><|lvr_end|>
    {post_visual_text_think[correct_idx]}
    {text_think}
  </think><answer>{answer}</answer>
"""

from __future__ import annotations

import json
import logging
import os
from functools import partial
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.utils import rank0_print
logger = logging.getLogger("LantErn-TetrisDataset")


class SFTTetrisDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        dummy: bool = False,
        use_lvr: bool = True,
        max_samples: int = None,
    ):
        super().__init__()
        self.processor = processor
        self.use_lvr = use_lvr
        data_dir = os.path.dirname(os.path.abspath(data_path))
        with open(data_path) as f:
            self.dataset = json.load(f)

        # TODO: remove this normalization once all datasets are regenerated with
        # relative paths (create_dataset.py now saves relative paths). This is a
        # temporary shim for existing JSON files that have absolute paths baked in
        # from the generation cluster (/e/project1/... or /mnt/scratch-nyx/...).
        def _normalize(path: str) -> str:
            marker = "analogy_data" + os.sep
            idx = path.find(marker)
            if idx != -1:
                path = path[idx + len(marker):]
            return os.path.join(data_dir, path)

        for sample in self.dataset:
            sample["img_path"] = _normalize(sample["img_path"])
            traces = sample["reasoning_traces"]
            traces["intermediate_img_path"] = _normalize(traces["intermediate_img_path"])

        rank0_print(f"Loaded {len(self.dataset)} analogy samples from {data_path} "
                    f"(use_lvr={use_lvr})", flush=True)

        if dummy:
            self.dataset = self.dataset[:1000]
        elif max_samples is not None:
            self.dataset = self.dataset[:max_samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        traces = data["reasoning_traces"]

        question = data["question"]
        answer   = traces["answer"]

        pre_text   = traces["pre_visual_text_think"]
        post_list  = traces["post_visual_text_think"]   # list of 4, one non-empty
        text_think = traces.get("text_think", "")
        inter_path = traces["intermediate_img_path"]

        post_text = next((t for t in post_list if t), "")

        composite_img = Image.open(data["img_path"])

        user_content = [
            {"type": "text",  "text": question},
            {"type": "image", "image": composite_img},
        ]

        if self.use_lvr:
            assistant_content = (
                "<think>"
                + pre_text
                + "<|lvr_start|><|lvr_sep|><|lvr_end|>"
                + post_text
                + text_think
                + "</think>"
                + "<answer>" + answer + "</answer>"
            )
            intermediate_img = Image.open(inter_path)
            return [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
                {"role": "assistant", "content": [{"type": "image", "image": intermediate_img}]},
            ]
        else:
            # NTP: pure text reasoning, no latent visual tokens
            assistant_content = (
                "<think>"
                + pre_text
                + post_text
                + text_think
                + "</think>"
                + "<answer>" + answer + "</answer>"
            )
            return [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
            ]


# ---------------------------------------------------------------------------
# Collate functions  (identical logic to sft_viscot_data.py)
# ---------------------------------------------------------------------------

def _mask_image_output_tokens(input_ids: torch.Tensor, image_token: int) -> torch.Tensor:
    return (input_ids == image_token).long()


def collate_fn_sft(samples: List[list], processor: AutoProcessor):
    latent_visuals = [s.pop(-1) for s in samples]

    text          = processor.apply_chat_template(samples, tokenize=False)
    image_inputs, video_inputs = process_vision_info(samples)

    # Expand <|lvr_sep|> to latent_size copies
    latent_text         = processor.apply_chat_template(latent_visuals, tokenize=False)
    latent_image_inputs, _ = process_vision_info(latent_visuals)
    latent_inputs = processor(
        text=latent_text,
        images=latent_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    latent_grid_thw = latent_inputs["image_grid_thw"]
    merge_length    = processor.image_processor.merge_size ** 2

    if processor.latent_size == -1:
        num_latent_tokens = [int(g.prod()) // merge_length for g in latent_grid_thw]
    else:
        num_latent_tokens = [processor.latent_size] * len(latent_grid_thw)

    lvr_sep = "<|lvr_sep|>"
    for i, n in enumerate(num_latent_tokens):
        if n > 1:
            text[i] = text[i].replace(lvr_sep, lvr_sep * n, 1)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding="max_length",
        max_length=4096,
        truncation=True,
    )

    # Build labels: only supervise the assistant turn
    labels = torch.full_like(inputs["input_ids"], -100)
    assistant_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_end   = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def _find(seq, subseq):
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return -1

    for i, ids in enumerate(inputs["input_ids"].tolist()):
        ts = _find(ids, assistant_start)
        te = ts + _find(ids[ts:], assistant_end)
        if ts == -1 or te == -1:
            continue
        s = ts + len(assistant_start)
        e = te + len(assistant_end) - 1
        labels[i, s:e] = torch.tensor(ids[s:e], dtype=torch.long)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.lvr_sep_id]             = -100

    inputs["labels"]          = labels
    inputs["latent_mask_out"] = _mask_image_output_tokens(inputs["input_ids"], processor.lvr_sep_id)
    inputs["latent_values"]   = latent_inputs["pixel_values"]
    inputs["latent_grid_thw"] = latent_grid_thw

    return inputs


def collate_fn_generate(samples: List[list], processor: AutoProcessor):
    latent_visuals = [s.pop(-1) for s in samples]
    user_samples   = [[s] for bs in samples for s in bs if s["role"] == "user"]
    labels         = [s for bs in samples for s in bs if s["role"] == "assistant"]
    labels         = [
        l["content"][0]["text"].split("<answer>")[-1].replace("</answer>", "")
        for l in labels
    ]

    image_inputs, video_inputs = process_vision_info(user_samples)
    text = processor.apply_chat_template(user_samples, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )

    latent_text         = processor.apply_chat_template(latent_visuals, tokenize=False)
    latent_image_inputs, _ = process_vision_info(latent_visuals)
    latent_inputs = processor(
        text=latent_text, images=latent_image_inputs,
        padding=True, return_tensors="pt",
    )
    inputs["latent_values"]   = latent_inputs["pixel_values"]
    inputs["latent_grid_thw"] = latent_inputs["image_grid_thw"]

    return inputs, labels


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

def make_tetris_data_module(
    processor: AutoProcessor,
    data_path: str,
    dummy: bool = False,
    generate: bool = False,
    use_lvr: bool = True,
    max_train_samples: int = None,
):
    """
    Load pre-split train/eval sets produced by create_dataset.py.
    data_path should point to train.json; eval.json is inferred from the same directory.
    use_lvr=False produces NTP-style batches (no latent visual tokens).
    """
    import os
    train_path = data_path  # expected: .../train.json
    eval_path  = os.path.join(os.path.dirname(data_path), "eval.json")

    train_ds = SFTTetrisDataset(data_path=train_path, processor=processor,
                                dummy=dummy, use_lvr=use_lvr, max_samples=max_train_samples)
    rank0_print(f"Train: {len(train_ds)} samples", flush=True)

    collate = collate_fn_sft if not generate else collate_fn_generate
    out = {
        "train_dataset": train_ds,
        "data_collator": partial(collate, processor=processor),
    }
    if os.path.exists(eval_path):
        eval_ds = SFTTetrisDataset(data_path=eval_path, processor=processor,
                                   dummy=dummy, use_lvr=use_lvr)
        rank0_print(f"Eval:  {len(eval_ds)} samples", flush=True)
        out["eval_dataset"] = eval_ds
    else:
        rank0_print(f"No eval.json found at {eval_path} — skipping eval split.", flush=True)
    return out
