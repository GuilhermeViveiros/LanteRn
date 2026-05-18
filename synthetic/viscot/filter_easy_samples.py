"""
Filter VisCoT MC test samples to keep only those that require the bbox crop.

A sample is KEPT only if:
  1. Qwen2.5-VL-32B answers WRONG with image + question alone (needs more info)
  2. Qwen2.5-VL-32B answers CORRECT with image + bbox_crop + question (bbox helps)

Everything else is discarded:
  - Correct without bbox  → too easy, doesn't need visual reasoning
  - Wrong both ways       → bbox doesn't help, unclear signal

Output: results/filter_easy/
  - keep_ids.json    → line indices of samples to keep
  - remove_ids.json  → line indices to remove, with reason

Submit via srun (requires GPU, 32B model needs ~70GB):
    srun --partition=h100 --qos=gpu-h100 --job-name=filter_easy --time=06:00:00 \
         --gpus-per-node=1 --tasks-per-node=1 --mem=80GB \
      bash -c 'export PYTHONPATH=/mnt/home/gviveiros/LantErn:$PYTHONPATH && \
               python -u synthetic/viscot/filter_easy_samples.py \
               > results/filter_easy.log 2>&1'
"""

import argparse
import json
import os
from functools import partial

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.utils import center_and_crop_image, extract_mc_answer

DATA_PATH = "/mnt/scratch-artemis/gviveiros/lantern/oe_to_mc/viscot_mc_test.jsonl"
MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"

system_content = "Put your final answer inside <answer>ANSWER_GOES_HERE</answer> tags."


def parse_options(options):
    if len(options) != 4:
        return None
    options = [opt.replace(opt.split(" ")[0] + " ", "") for opt in options]
    if any(opt is None for opt in options):
        return None
    return options


class VisCoTDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        invalid = 0
        with open(data_path) as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                options = parse_options(sample["options"])
                if options is None:
                    invalid += 1
                    continue
                self.samples.append({
                    "idx": idx,
                    "question": sample["question"] + "\nOptions:\n" + "\n".join(options),
                    "options": options,
                    "label": sample["answer"],
                    "img_path": sample["img_path"],
                    "bbox": sample["bbox"],
                })
        print(f"Loaded {len(self.samples)} samples ({invalid} skipped, invalid options)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["img_path"])
        bbox = s["bbox"][0] if isinstance(s["bbox"][0], list) else s["bbox"]
        crop = center_and_crop_image(img, bbox)
        return {**s, "image": img, "crop": crop}


def collate_fn(batch, processor, include_crop: bool):
    messages, labels, options, indices = [], [], [], []
    for sample in batch:
        indices.append(sample["idx"])
        labels.append(sample["label"])
        options.append(sample["options"])
        content = [{"type": "image", "image": sample["image"]}]
        if include_crop:
            content.append({"type": "image", "image": sample["crop"]})
        content.append({"type": "text", "text": sample["question"]})
        messages.append([
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
            {"role": "user",   "content": content},
        ])

    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, labels, options, indices


@torch.no_grad()
def run_inference(model, processor, inputs):
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default=DATA_PATH)
    parser.add_argument("--model_id",   type=str, default=MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/filter_easy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model:  {args.model_id}")
    print(f"Data:   {args.data_path}")
    print(f"Output: {args.output_dir}")

    # Load base Qwen2.5-VL-32B — no LantErn patches
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    min_pixels = 256 * 28 * 28
    max_pixels = 3500 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.padding_side = "left"

    dataset = VisCoTDataset(args.data_path)
    loader_no_crop   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=False))
    loader_with_crop = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=True))

    ckpt_pass1 = os.path.join(args.output_dir, "checkpoint_pass1.json")

    # Pass 1: without bbox (resume if checkpoint exists)
    print("\n--- Pass 1: image + question (no bbox) ---")
    if os.path.exists(ckpt_pass1):
        print(f"Resuming from checkpoint: {ckpt_pass1}")
        with open(ckpt_pass1) as f:
            results_no_crop = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(results_no_crop)} pass-1 results from checkpoint")
    else:
        results_no_crop = {}  # idx -> correct (bool)
        for inputs, labels, options_batch, indices in tqdm(loader_no_crop, desc="No bbox"):
            decoded = run_inference(model, processor, inputs)
            answers = [extract_mc_answer(x, opts) for x, opts in zip(decoded, options_batch)]
            for idx, ans, label in zip(indices, answers, labels):
                results_no_crop[int(idx)] = (ans == label)
        with open(ckpt_pass1, "w") as f:
            json.dump({str(k): v for k, v in results_no_crop.items()}, f)
        print(f"Pass-1 checkpoint saved to {ckpt_pass1}")

    n_correct_without = sum(results_no_crop.values())
    print(f"Correct without bbox: {n_correct_without}/{len(results_no_crop)} ({n_correct_without/len(results_no_crop):.1%})")

    # Pass 2: with bbox crop — only for samples wrong in pass 1
    print("\n--- Pass 2: image + bbox_crop + question ---")
    need_crop_check = {idx for idx, correct in results_no_crop.items() if not correct}
    results_with_crop = {}  # idx -> correct (bool)

    for inputs, labels, options_batch, indices in tqdm(loader_with_crop, desc="With bbox"):
        if not any(int(idx) in need_crop_check for idx in indices):
            continue
        decoded = run_inference(model, processor, inputs)
        answers = [extract_mc_answer(x, opts) for x, opts in zip(decoded, options_batch)]
        for idx, ans, label in zip(indices, answers, labels):
            if int(idx) in need_crop_check:
                results_with_crop[int(idx)] = (ans == label)

    # Classify
    keep_ids, remove_easy, remove_hard = [], [], []
    for idx in results_no_crop:
        wrong_without = not results_no_crop[idx]
        right_with    = results_with_crop.get(idx, False)
        if wrong_without and right_with:
            keep_ids.append(idx)       # bbox is decisive → keep
        elif not wrong_without:
            remove_easy.append(idx)    # correct without bbox → too easy
        else:
            remove_hard.append(idx)    # wrong both ways → bbox doesn't help

    total = len(results_no_crop)
    print("\n=== Results ===")
    print(f"Keep (bbox decisive):        {len(keep_ids):5d} / {total} ({len(keep_ids)/total:.1%})")
    print(f"Remove (easy, no bbox):      {len(remove_easy):5d} / {total} ({len(remove_easy)/total:.1%})")
    print(f"Remove (hard, bbox no help): {len(remove_hard):5d} / {total} ({len(remove_hard)/total:.1%})")

    with open(os.path.join(args.output_dir, "keep_ids.json"), "w") as f:
        json.dump({"keep_ids": sorted(keep_ids), "n_keep": len(keep_ids), "total": total}, f, indent=2)

    with open(os.path.join(args.output_dir, "remove_ids.json"), "w") as f:
        json.dump({
            "remove_easy": sorted(remove_easy),
            "remove_hard": sorted(remove_hard),
            "total_removed": len(remove_easy) + len(remove_hard),
            "total": total,
        }, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")
