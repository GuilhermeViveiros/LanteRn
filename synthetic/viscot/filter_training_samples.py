"""
Filter LantErn SFT training samples to keep only those that require the bbox crop.

A sample is KEPT only if:
  1. Qwen2.5-VL-32B answers WRONG with image + question alone
  2. Qwen2.5-VL-32B answers CORRECT with image + bbox_crop + question

Correctness is judged by the same model: given (question, ground_truth, prediction),
it returns a score 0-10. Score >= threshold → correct.

Everything else is discarded:
  - Correct without bbox  → too easy, doesn't need visual reasoning
  - Wrong both ways       → bbox doesn't help, unclear signal

Output: results/filter_training/
  - keep_ids.json         → indices compatible with SFTDataset filter_ids_path
  - remove_ids.json       → removed indices with reason
  - pass1_checkpoint.json → pass-1 results (auto-resumes on re-run)

Flags:
  --dummy N      Only process the first N samples (quick sanity check)
  --inspect N    Print N samples and exit without running inference

Run from the repo root with -m (no PYTHONPATH needed):

Submit via srun (32B model needs ~70GB VRAM):
    srun --partition=h100 --qos=gpu-h100 --job-name=filter-train --time=12:00:00 \\
         --gpus-per-node=1 --tasks-per-node=1 --mem=80GB \\
      bash -c 'cd /mnt/home/gviveiros/LantErn && \\
               python -u -m synthetic.viscot.filter_training_samples \\
               > results/filter_training.log 2>&1'

For a quick dummy run (first 50 samples, verbose per-sample output):
    cd /mnt/home/gviveiros/LantErn
    python -m synthetic.viscot.filter_training_samples --dummy 50

For inspect (no GPU needed):
    python -m synthetic.viscot.filter_training_samples --inspect 10
"""

import json
import os
import argparse
import torch
from tqdm import tqdm
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import center_and_crop_image

DATA_PATH = "/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"
# MODEL_ID  = "Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct" # debug purpouses

ANSWER_SYSTEM = "Answer the question concisely. Put your final answer inside <answer>ANSWER_GOES_HERE</answer> tags."

JUDGE_SYSTEM = (
    "You are a strict answer grader. Given a question, a ground-truth answer, and a model's "
    "predicted answer, decide if they are semantically equivalent. Reply with a single integer "
    "score from 0 to 10, where 10 means perfectly correct and 0 means completely wrong. "
    "Output only the integer, nothing else."
)

JUDGE_THRESHOLD = 7  # score >= threshold → correct


class TrainingDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = None):
        with open(data_path, "r") as f:
            raw = json.load(f)

        # Apply the same pre-validation as SFTDataset
        self.samples = []
        skipped = 0
        for orig_idx, item in enumerate(raw):
            if len(item["bboxs"]) > 1:
                skipped += 1
                continue
            if "34084d4c3c347b83.jpg" in item.get("img_path", ""):
                skipped += 1
                continue
            self.samples.append({"orig_idx": orig_idx, **item})

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Map orig_idx → position in self.samples for fast lookup
        self.orig_to_pos = {s["orig_idx"]: i for i, s in enumerate(self.samples)}

        print(f"Loaded {len(self.samples)} samples ({skipped} skipped: >1 bbox or bad img)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["img_path"]).convert("RGB")
        crop = center_and_crop_image(img, s["bboxs"][0])
        return {**s, "image": img, "crop": crop}


def collate_fn(batch, processor, include_crop: bool):
    messages, ground_truths, orig_indices, questions = [], [], [], []
    for sample in batch:
        orig_indices.append(sample["orig_idx"])
        ground_truths.append(sample["answer"])
        questions.append(sample["question"])
        content = [{"type": "image", "image": sample["image"]}]
        if include_crop:
            content.append({"type": "image", "image": sample["crop"]})
        content.append({"type": "text", "text": sample["question"]})
        messages.append([
            {"role": "system", "content": [{"type": "text", "text": ANSWER_SYSTEM}]},
            {"role": "user",   "content": content},
        ])
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt")
    return inputs, ground_truths, orig_indices, questions


@torch.no_grad()
def run_inference(model, processor, inputs):
    inputs = inputs.to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)


@torch.no_grad()
def judge_answers(model, processor, questions: list, generated: list, ground_truths: list) -> list[int]:
    """
    Ask the same model to score each (question, ground_truth, prediction) triple.
    Returns integer scores 0-10 per sample.
    """
    messages = []
    for q, gen, gt in zip(questions, generated, ground_truths):
        pred = gen.split("<answer>")[-1].split("</answer>")[0].strip()
        messages.append([
            {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM}]},
            {"role": "user",   "content": [{"type": "text", "text":
                f"Question: {q}\nGround truth: {gt}\nModel answer: {pred}"
            }]},
        ])
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    scores = []
    for raw in decoded:
        try:
            scores.append(int(raw.strip()))
        except ValueError:
            scores.append(0)  # unparseable → treat as wrong
    return scores


def inspect_samples(dataset, n: int):
    """Print n samples without running any inference."""
    print(f"\n{'='*60}")
    print(f"INSPECT MODE — {min(n, len(dataset))} samples")
    print(f"{'='*60}")
    for i in range(min(n, len(dataset))):
        s = dataset.samples[i]
        print(f"\n[{i}] orig_idx={s['orig_idx']}  dataset={s.get('dataset', '?')}")
        print(f"  question  : {s['question']}")
        print(f"  answer    : {s['answer']}")
        print(f"  rt answer : {s['reasoning_traces'].get('answer', 'N/A')}")
        print(f"  bbox      : {s['bboxs'][0]}")
        print(f"  img_path  : {s['img_path']}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default=DATA_PATH)
    parser.add_argument("--model_id",   type=str, default=MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/filter_training")
    parser.add_argument("--threshold",  type=int, default=JUDGE_THRESHOLD,
                        help="Judge score threshold 0-10 (>= threshold → correct)")
    parser.add_argument("--dummy",      type=int, default=None, metavar="N",
                        help="Only process first N samples (sanity check)")
    parser.add_argument("--inspect",    type=int, default=None, metavar="N",
                        help="Print N samples and exit without running inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = TrainingDataset(args.data_path, max_samples=args.dummy)

    if args.inspect is not None:
        inspect_samples(dataset, args.inspect)
        exit(0)

    print(f"Model:           {args.model_id}")
    print(f"Data:            {args.data_path}")
    print(f"Samples:         {len(dataset)}")
    print(f"Output:          {args.output_dir}")
    print(f"Judge threshold: {args.threshold}/10")
    if args.dummy:
        print(f"*** DUMMY MODE: first {args.dummy} samples only ***")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    min_pixels = 256 * 28 * 28
    max_pixels = 3500 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"

    loader_no_crop   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=False))
    loader_with_crop = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=True))

    ckpt_pass1 = os.path.join(args.output_dir, "pass1_checkpoint.json")

    # ── Pass 1: image + question only ─────────────────────────────────────────
    print("\n--- Pass 1: image + question (no bbox) ---")
    if os.path.exists(ckpt_pass1) and args.dummy is None:
        print(f"Resuming from checkpoint: {ckpt_pass1}")
        with open(ckpt_pass1) as f:
            results_no_crop = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(results_no_crop)} pass-1 results")
    else:
        results_no_crop = {}
        for inputs, ground_truths, orig_indices, questions in tqdm(loader_no_crop, desc="No bbox"):
            decoded = run_inference(model, processor, inputs)
            scores  = judge_answers(model, processor, questions, decoded, ground_truths)
            for idx, gen, gt, q, score in zip(orig_indices, decoded, ground_truths, questions, scores):
                correct = score >= args.threshold
                results_no_crop[int(idx)] = {"correct": correct, "score": score, "generated": gen[:200], "gt": gt}
                if args.dummy:
                    pred = gen.split("<answer>")[-1].split("</answer>")[0].strip()
                    print(f"  [no-crop] idx={idx:5d}  score={score:2d}  gt={gt!r:30s}  pred={pred!r:30s}")

        if args.dummy is None:
            with open(ckpt_pass1, "w") as f:
                json.dump({str(k): v for k, v in results_no_crop.items()}, f)
            print(f"Pass-1 checkpoint saved → {ckpt_pass1}")

    n_correct = sum(v["correct"] for v in results_no_crop.values())
    total = len(results_no_crop)
    print(f"Correct without bbox: {n_correct}/{total} ({n_correct/total:.1%})")

    # ── Pass 2: image + bbox_crop + question (only for pass-1 failures) ───────
    print("\n--- Pass 2: image + bbox_crop + question ---")
    need_crop_check = {idx for idx, v in results_no_crop.items() if not v["correct"]}
    print(f"Samples needing pass-2: {len(need_crop_check)}")

    results_with_crop = {}
    for inputs, ground_truths, orig_indices, questions in tqdm(loader_with_crop, desc="With bbox"):
        if not any(int(idx) in need_crop_check for idx in orig_indices):
            continue
        decoded = run_inference(model, processor, inputs)
        scores  = judge_answers(model, processor, questions, decoded, ground_truths)
        for idx, gen, gt, q, score in zip(orig_indices, decoded, ground_truths, questions, scores):
            if int(idx) in need_crop_check:
                correct = score >= args.threshold
                results_with_crop[int(idx)] = {"correct": correct, "score": score, "generated": gen[:200], "gt": gt}
                if args.dummy:
                    pred = gen.split("<answer>")[-1].split("</answer>")[0].strip()
                    print(f"  [w-crop]  idx={idx:5d}  score={score:2d}  gt={gt!r:30s}  pred={pred!r:30s}")

    # ── Classify ──────────────────────────────────────────────────────────────
    keep_ids, remove_easy, remove_hard = [], [], []
    for idx, res in results_no_crop.items():
        wrong_without = not res["correct"]
        right_with    = results_with_crop.get(idx, {}).get("correct", False)
        if wrong_without and right_with:
            keep_ids.append(idx)
        elif not wrong_without:
            remove_easy.append(idx)
        else:
            remove_hard.append(idx)

    print(f"\n=== Results ===")
    print(f"Keep (bbox decisive):        {len(keep_ids):6d} / {total} ({len(keep_ids)/total:.1%})")
    print(f"Remove (easy, no bbox):      {len(remove_easy):6d} / {total} ({len(remove_easy)/total:.1%})")
    print(f"Remove (hard, bbox no help): {len(remove_hard):6d} / {total} ({len(remove_hard)/total:.1%})")

    suffix = f"_dummy{args.dummy}" if args.dummy else ""
    with open(os.path.join(args.output_dir, f"keep_ids{suffix}.json"), "w") as f:
        json.dump({"keep_ids": sorted(keep_ids), "n_keep": len(keep_ids), "total": total}, f, indent=2)
    with open(os.path.join(args.output_dir, f"remove_ids{suffix}.json"), "w") as f:
        json.dump({
            "remove_easy": sorted(remove_easy),
            "remove_hard": sorted(remove_hard),
            "total_removed": len(remove_easy) + len(remove_hard),
            "total": total,
        }, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")
