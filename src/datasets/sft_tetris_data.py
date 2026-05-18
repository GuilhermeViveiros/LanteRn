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

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor

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
        grayscale_intermediate: bool = False,
    ):
        super().__init__()
        self.processor = processor
        self.use_lvr = use_lvr
        self.grayscale_intermediate = grayscale_intermediate
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

    def get_group_key(self, idx: int, key: str) -> str:
        """Return a raw metadata field without loading any images (for family batching)."""
        return self.dataset[idx].get(key, "unknown")

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
                + "<answer> " + answer + "</answer>"
            )
            if self.grayscale_intermediate:
                intermediate_img = Image.open(inter_path).convert("RGB")
            else:
                intermediate_img = Image.open(inter_path).convert("RGB")
            latent_msg = {"role": "assistant", "content": [{"type": "image", "image": intermediate_img}],
                          "shape_C_name": data.get("shape_C_name"),
                          "intermediate_key": data.get("intermediate_key", "")}
            return [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
                latent_msg,
            ]
        else:
            # NTP: pure text reasoning, no latent visual tokens
            assistant_content = (
                "<think>"
                + pre_text
                + post_text
                + text_think
                + "</think>"
                + "<answer> " + answer + "</answer>"
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


def collate_fn_latent_sft(samples: list[list], processor: AutoProcessor):
    # Extract shape_C_name before popping the latent message
    shape_names = [s[-1].pop("shape_C_name", None) for s in samples]
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

    # In text_think the answer appears as "option (X)" →  ' ('(320) + bare_letter(64-67).
    # We find that position for answer_ce_loss monitoring.
    _OPEN_PAREN = 320          # token for ' ('
    _LETTER_IDS = {64, 65, 66, 67}  # bare a / b / c / d
    answer_positions = torch.full((len(inputs["input_ids"]),), -1, dtype=torch.long)

    for i, ids in enumerate(inputs["input_ids"].tolist()):
        ts = _find(ids, assistant_start)
        te = ts + _find(ids[ts:], assistant_end)
        if ts == -1 or te == -1:
            continue
        s = ts + len(assistant_start)
        e = te + len(assistant_end) - 1
        labels[i, s:e] = torch.tensor(ids[s:e], dtype=torch.long)

        # Find "option (X)" pattern: ' (' followed by bare letter in supervised region
        for t in range(s, e - 1):
            if ids[t] == _OPEN_PAREN and ids[t + 1] in _LETTER_IDS:
                answer_positions[i] = t + 1
                break

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.lvr_sep_id]             = -100

    import zlib
    inputs["labels"]           = labels
    inputs["answer_positions"] = answer_positions
    inputs["latent_mask_out"]  = _mask_image_output_tokens(inputs["input_ids"], processor.lvr_sep_id)
    inputs["latent_values"]    = latent_inputs["pixel_values"]
    inputs["latent_grid_thw"]  = latent_grid_thw
    # Encode shape names as deterministic int64 so Accelerate can concatenate across
    # gradient-accumulation steps (strings can't be tensor-concatenated).
    inputs["shape_name_ids"] = torch.tensor([
        zlib.adler32(n.encode()) if n is not None else -1
        for n in shape_names
    ], dtype=torch.long)

    return inputs


def collate_fn_ntp(samples: list[list], processor: AutoProcessor):
    """Collate for NTP (use_lvr=False): no latent visual, pure text reasoning."""
    text = processor.apply_chat_template(samples, tokenize=False)
    image_inputs, video_inputs = process_vision_info(samples)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding="max_length",
        max_length=4096,
        truncation=True,
    )

    labels = torch.full_like(inputs["input_ids"], -100)
    assistant_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_end   = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def _find(seq, subseq):
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return -1

    _OPEN_PAREN = 320
    _LETTER_IDS = {64, 65, 66, 67}
    answer_positions = torch.full((len(inputs["input_ids"]),), -1, dtype=torch.long)

    for i, ids in enumerate(inputs["input_ids"].tolist()):
        ts = _find(ids, assistant_start)
        te = ts + _find(ids[ts:], assistant_end)
        if ts == -1 or te == -1:
            continue
        s = ts + len(assistant_start)
        e = te + len(assistant_end) - 1
        labels[i, s:e] = torch.tensor(ids[s:e], dtype=torch.long)

        for t in range(s, e - 1):
            if ids[t] == _OPEN_PAREN and ids[t + 1] in _LETTER_IDS:
                answer_positions[i] = t + 1
                break

    labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"]           = labels
    inputs["answer_positions"] = answer_positions

    return inputs


def collate_fn_generate_ntp(samples: list[list], processor: AutoProcessor):
    """Generate collate for NTP mode: no latent visual, returns (inputs, labels)."""
    user_samples = [[s] for bs in samples for s in bs if s["role"] == "user"]
    labels       = [s for bs in samples for s in bs if s["role"] == "assistant"]
    labels       = [
        l["content"][0]["text"].split("<answer>")[-1].replace("</answer>", "")
        for l in labels
    ]

    image_inputs, video_inputs = process_vision_info(user_samples)
    text = processor.apply_chat_template(user_samples, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )
    return inputs, labels


def collate_fn_generate(samples: list[list], processor: AutoProcessor):
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
    grayscale_intermediate: bool = False,
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
                                dummy=dummy, use_lvr=use_lvr, max_samples=max_train_samples,
                                grayscale_intermediate=grayscale_intermediate)
    rank0_print(f"Train: {len(train_ds)} samples (grayscale_intermediate={grayscale_intermediate})", flush=True)

    if generate:
        collate = collate_fn_generate
    elif use_lvr:
        collate = collate_fn_latent_sft
    else:
        collate = collate_fn_ntp
    out = {
        "train_dataset": train_ds,
        "data_collator": partial(collate, processor=processor),
    }
    if os.path.exists(eval_path):
        eval_ds = SFTTetrisDataset(data_path=eval_path, processor=processor,
                                   dummy=dummy, use_lvr=use_lvr,
                                   grayscale_intermediate=grayscale_intermediate)
        rank0_print(f"Eval:  {len(eval_ds)} samples", flush=True)
        out["eval_dataset"] = eval_ds
    else:
        rank0_print(f"No eval.json found at {eval_path} — skipping eval split.", flush=True)
    return out
