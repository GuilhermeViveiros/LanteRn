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

Submit via srun — single GPU (32B model needs ~70GB VRAM):
    srun --partition=h100 --qos=gpu-h100 --job-name=filter-train --time=12:00:00 \\
         --gpus-per-node=1 --tasks-per-node=1 --mem=80GB \\
      bash -c 'cd /mnt/home/gviveiros/LantErn && HF_HUB_OFFLINE=1 \\
               python -u -m synthetic.viscot.filter_training_samples \\
               > results/filter_training.log 2>&1'

Submit via srun — multi-GPU (4x H100, ~4x throughput via data parallelism):
    srun --partition=h100 --qos=gpu-h100 --job-name=filter-train --time=06:00:00 \\
         --gpus-per-node=4 --tasks-per-node=1 --mem=320GB \\
      bash -c 'cd /mnt/home/gviveiros/LantErn && HF_HUB_OFFLINE=1 \\
               torchrun --nproc_per_node=4 -m synthetic.viscot.filter_training_samples \\
               > results/filter_training.log 2>&1'

Note: with 32B model each GPU needs ~70GB. If OOM, reduce --batch_size.

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
import torch.distributed as dist
from tqdm import tqdm
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import center_and_crop_image

DATA_PATH = "/e/project1/jureap126/gviveiros/lantern/LantErn_VisCot_data.json"
MODEL_ID  = "Qwen/Qwen2.5-VL-32B-Instruct"
# MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct" # debug purpouses

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
            # replace img_path str from /mnt/data-artemis/gviveiros/lantern/ to /e/project1/jureap126/gviveiros/lantern
            
            item = {**item, "img_path": item["img_path"].replace(
                    "/mnt/data-artemis/gviveiros/lantern/",
                    "/e/project1/jureap126/gviveiros/lantern/"
            )}
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
    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, default=DATA_PATH)
    parser.add_argument("--model_id",   type=str, default=MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/filter_training")
    parser.add_argument("--threshold",  type=int, default=JUDGE_THRESHOLD,
                        help="Judge score threshold 0-10 (>= threshold → correct)")
    parser.add_argument("--dummy",      type=int, default=None, metavar="N",
                        help="Only process first N samples (sanity check)")
    parser.add_argument("--inspect",    type=int, default=None, metavar="N",
                        help="Print N samples and exit without running inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dummy_tag = f"_dummy{args.dummy}" if args.dummy else ""

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    rank        = int(os.environ.get("RANK", local_rank))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main  = rank == 0
    rank_tag = f"_rank{rank}" if distributed else ""

    dataset = TrainingDataset(args.data_path, max_samples=args.dummy)

    if args.inspect is not None:
        if is_main:
            inspect_samples(dataset, args.inspect)
        if distributed:
            dist.destroy_process_group()
        exit(0)

    if is_main:
        print(f"Model:           {args.model_id}")
        print(f"Data:            {args.data_path}")
        print(f"Samples (total): {len(dataset)}")
        print(f"Output:          {args.output_dir}")
        print(f"Judge threshold: {args.threshold}/10")
        if distributed:
            print(f"Distributed:     {world_size} ranks, ~{len(dataset) // world_size} samples/rank")
        if args.dummy:
            print(f"*** DUMMY MODE: first {args.dummy} samples only ***")

    device_map = {"": f"cuda:{local_rank}"} if distributed else "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=device_map,
        local_files_only=True,
    )
    model.eval()

    min_pixels = 256 * 28 * 28
    max_pixels = 3500 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels,
                                              local_files_only=True)
    processor.tokenizer.padding_side = "left"

    sampler_kwargs = dict(sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)) \
                     if distributed else {}
    loader_no_crop   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=False),
                                  **sampler_kwargs)
    loader_with_crop = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=partial(collate_fn, processor=processor, include_crop=True),
                                  **sampler_kwargs)

    # ── Pass 1: image + question only ─────────────────────────────────────────
    print(f"[rank {rank}] --- Pass 1: image + question (no bbox) ---")
    results_no_crop = {}
    for inputs, ground_truths, orig_indices, questions in tqdm(
        loader_no_crop, desc=f"No bbox (rank {rank})", disable=not is_main
    ):
        decoded = run_inference(model, processor, inputs)
        scores  = judge_answers(model, processor, questions, decoded, ground_truths)
        for idx, gen, gt, score in zip(orig_indices, decoded, ground_truths, scores):
            correct = score >= args.threshold
            results_no_crop[int(idx)] = {"correct": correct, "score": score, "generated": gen[:200], "gt": gt}
            if args.dummy:
                pred = gen.split("<answer>")[-1].split("</answer>")[0].strip()
                print(f"  [no-crop] idx={idx:5d}  score={score:2d}  gt={gt!r:30s}  pred={pred!r:30s}")

    total     = len(results_no_crop)
    n_correct = sum(v["correct"] for v in results_no_crop.values())
    print(f"[rank {rank}] Correct without bbox: {n_correct}/{total} ({n_correct/total:.1%})")

    # ── Pass 2: image + bbox_crop + question (only for pass-1 failures) ───────
    print(f"[rank {rank}] --- Pass 2: image + bbox_crop + question ---")
    need_crop_check = {idx for idx, v in results_no_crop.items() if not v["correct"]}
    print(f"[rank {rank}] Samples needing pass-2: {len(need_crop_check)}")

    results_with_crop = {}
    for inputs, ground_truths, orig_indices, questions in tqdm(
        loader_with_crop, desc=f"With bbox (rank {rank})", disable=not is_main
    ):
        if not any(int(idx) in need_crop_check for idx in orig_indices):
            continue
        decoded = run_inference(model, processor, inputs)
        scores  = judge_answers(model, processor, questions, decoded, ground_truths)
        for idx, gen, gt, score in zip(orig_indices, decoded, ground_truths, scores):
            if int(idx) in need_crop_check:
                correct = score >= args.threshold
                results_with_crop[int(idx)] = {"correct": correct, "score": score, "generated": gen[:200], "gt": gt}
                if args.dummy:
                    pred = gen.split("<answer>")[-1].split("</answer>")[0].strip()
                    print(f"  [w-crop]  idx={idx:5d}  score={score:2d}  gt={gt!r:30s}  pred={pred!r:30s}")

    # ── Classify and save per-rank partial ────────────────────────────────────
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

    print(f"[rank {rank}] Keep: {len(keep_ids)}  Easy: {len(remove_easy)}  Hard: {len(remove_hard)}")

    partial_path = os.path.join(args.output_dir, f"partial{rank_tag}{dummy_tag}.json")
    with open(partial_path, "w") as f:
        json.dump({
            "keep_ids":    sorted(keep_ids),
            "remove_easy": sorted(remove_easy),
            "remove_hard": sorted(remove_hard),
            "total":       total,
        }, f, indent=2)
    print(f"[rank {rank}] Partial saved → {partial_path}")

    if distributed:
        dist.barrier()

    if is_main:
        all_keep, all_easy, all_hard, all_total = [], [], [], 0
        for r in range(world_size):
            r_tag = f"_rank{r}" if distributed else ""
            p = os.path.join(args.output_dir, f"partial{r_tag}{dummy_tag}.json")
            with open(p) as f:
                d = json.load(f)
            all_keep.extend(d["keep_ids"])
            all_easy.extend(d["remove_easy"])
            all_hard.extend(d["remove_hard"])
            all_total += d["total"]

        print(f"\n=== Final Results ({all_total} samples) ===")
        print(f"Keep (bbox decisive):        {len(all_keep):6d} / {all_total} ({len(all_keep)/all_total:.1%})")
        print(f"Remove (easy, no bbox):      {len(all_easy):6d} / {all_total} ({len(all_easy)/all_total:.1%})")
        print(f"Remove (hard, bbox no help): {len(all_hard):6d} / {all_total} ({len(all_hard)/all_total:.1%})")

        with open(os.path.join(args.output_dir, f"keep_ids{dummy_tag}.json"), "w") as f:
            json.dump({"keep_ids": sorted(all_keep), "n_keep": len(all_keep), "total": all_total}, f, indent=2)
        with open(os.path.join(args.output_dir, f"remove_ids{dummy_tag}.json"), "w") as f:
            json.dump({
                "remove_easy":   sorted(all_easy),
                "remove_hard":   sorted(all_hard),
                "total_removed": len(all_easy) + len(all_hard),
                "total":         all_total,
            }, f, indent=2)
        print(f"Saved to {args.output_dir}/")

    if distributed:
        dist.destroy_process_group()
