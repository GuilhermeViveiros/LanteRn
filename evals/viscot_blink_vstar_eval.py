"""
VisCoT / Blink / VStar accuracy evaluation for LantErn models with ablation conditions.

Conditions:
  own        - model generates its own latent tokens (self-feeding hidden-state loop)
  no_lvr     - no latent block; model answers directly from question+image
  zeros      - inject zero embeddings at latent token positions
  gt         - inject GT bbox crop embeddings (VisCoT only)
  random_bbox - inject random-location crop (same bbox size on VisCoT; random 20-60% crop on Blink/VStar)

Usage:
  # Sanity check (5 samples)
  python -m evals.viscot_blink_vstar_eval \\
      --model_ref /path/to/checkpoint \\
      --conditions own no_lvr zeros \\
      --benchmarks viscot \\
      --max_samples 5 \\
      --output_dir results/lantern_ablation_debug

  # Full run
  srun --partition=h100 --qos=gpu-h100 --job-name=lantern_eval --time=06:00:00 \\
       --gpus-per-node=1 --tasks-per-node=1 --mem=50GB \\
    bash -c 'export PYTHONPATH=/path/to/LantErn:$PYTHONPATH && \\
             python -u -m evals.viscot_blink_vstar_eval \\
               --model_ref /mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1500 \\
               --benchmarks blink vstar \\
               --output_dir results/lantern_ablation \\
               > results/lantern_ablation.log 2>&1'
"""

import argparse
import json
import os
import random
import string
from functools import partial
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from datasets import load_dataset

# /mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/
# /mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-1500/
from src.constants import SCRATCH_ARTEMIS, VISCOT_MC_TEST_PATH
from src.lantern_generate.generate import generate as lantern_generate
from src.lantern_generate.generate import generate_skip_latent
from src.models import load_model
from src.models.utils import apply_latent_compression
from src.train import set_latent_tokens
from src.utils import center_and_crop_image, extract_mc_answer

VISCOT_TEST_PATH = VISCOT_MC_TEST_PATH
IMG_ROOT         = SCRATCH_ARTEMIS + "/"
BLINK_CATEGORIES = ["Object_Localization", "Spatial_Relation"]
LATENT_SIZE      = 8


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_cache(path: str) -> list:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_cache(path: str, results: list):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _cache_complete(path: str, max_samples: Optional[int]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return False
    if not data:
        return False
    if max_samples is not None and len(data) < max_samples:
        return False
    return True


# ── Latent embedding helpers ───────────────────────────────────────────────────

@torch.no_grad()
def encode_crop(model, processor, crop: Image.Image, device) -> torch.Tensor:
    """Encode a PIL crop → (1, latent_size, hidden_size)."""
    inputs   = processor.image_processor(images=[crop], return_tensors="pt")
    pix      = inputs["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
    grid_thw = inputs["image_grid_thw"].to(device)
    return apply_latent_compression(model, pix, grid_thw, model.config.latent_size)


def get_gt_latent_embeds(model, processor, img_path: str, bbox: list, device) -> torch.Tensor:
    img  = Image.open(img_path).convert("RGB")
    crop = center_and_crop_image(img, bbox)
    return encode_crop(model, processor, crop, device)


def get_random_bbox_latent_embeds(model, processor, img_path_or_img, bbox: list,
                                  device) -> torch.Tensor:
    """Same bbox size, random location — for VisCoT random_bbox condition."""
    if isinstance(img_path_or_img, str):
        img = Image.open(img_path_or_img).convert("RGB")
    else:
        img = img_path_or_img
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    W, H   = img.width, img.height
    rx1 = random.randint(0, max(0, W - bw))
    ry1 = random.randint(0, max(0, H - bh))
    crop = center_and_crop_image(img, [rx1, ry1, rx1 + bw, ry1 + bh])
    return encode_crop(model, processor, crop, device)


def get_random_crop_latent_embeds(model, processor, img, device) -> torch.Tensor:
    """Random 20-60% crop — for Blink/VStar random_bbox condition."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    W, H = img.width, img.height
    cw = random.randint(max(1, int(W * 0.2)), max(1, int(W * 0.6)))
    ch = random.randint(max(1, int(H * 0.2)), max(1, int(H * 0.6)))
    x1 = random.randint(0, max(0, W - cw))
    y1 = random.randint(0, max(0, H - ch))
    crop = img.crop((x1, y1, x1 + cw, y1 + ch))
    return encode_crop(model, processor, crop, device)


def get_zeros_latent_embeds(model, device) -> torch.Tensor:
    """Zero embeddings — (1, latent_size, hidden_size)."""
    return torch.zeros(
        1, model.config.latent_size, model.config.hidden_size,
        dtype=next(model.parameters()).dtype, device=device,
    )


# ── Core inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, processor, images, question: str, condition: str,
                  device, gt_latent_embeds: Optional[torch.Tensor] = None,
                  max_new_tokens: int = 512, perturbation: str = None) -> str:
    if not isinstance(images, list):
        images = [images]

    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": (Image.open(img).convert("RGB") if isinstance(img, str) else img)}
          for img in images],
        {"type": "text", "text": question},
    ]}]

    text         = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs       = processor(text=[text], images=image_inputs, padding=True,
                             return_tensors="pt").to(device)

    prompt_len = inputs["input_ids"].shape[1]

    if condition in ("no_lvr", "empty_latent"):
        custom_gen = partial(generate_skip_latent)
    else:
        custom_gen = partial(lantern_generate, gt_latent_embeds=gt_latent_embeds,
                             perturbation=perturbation)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        custom_generate=custom_gen,
        use_cache=True,
        return_dict_in_generate=False,
        output_attentions=False,
    )

    generated = output_ids[0, prompt_len:]
    decoded   = processor.decode(generated, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
    n_lat = int((generated == model.config.lvr_start_id).sum())
    print(f"  >> lat_blocks={n_lat} toks={generated.shape[0]} | {decoded}")
    return decoded


# ── VisCoT ─────────────────────────────────────────────────────────────────────

def load_viscot(path: str, img_root: str, max_samples: Optional[int] = None) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            options = s.get("options", [])
            if len(options) != 4:
                continue
            img_path = s.get("img_path", s.get("image", ""))
            img_path = img_path.replace("/mnt/data-artemis/gviveiros/lantern/", img_root)
            img_path = img_path.replace("/mnt/scratch-artemis/gviveiros/lantern/", img_root)
            bbox = s.get("bbox") or s.get("bboxs")
            if isinstance(bbox, list) and isinstance(bbox[0], list):
                bbox = bbox[0]
            options_clean = [o.split(" ", 1)[1] if " " in o else o for o in options]
            question = s["question"] + "\nOptions:\n" + "\n".join(options)
            data.append({
                "img_path": img_path,
                "question": question,
                "options":  options_clean,
                "label":    s.get("answer", ""),
                "bbox":     bbox,
            })
    return data[:max_samples] if max_samples else data


def evaluate_viscot(model, processor, condition: str, out_dir: str, device,
                    viscot_path: str = VISCOT_TEST_PATH, img_root: str = IMG_ROOT,
                    max_samples: Optional[int] = None, corrupt_image: bool = False) -> dict:
    if condition == "gt" and False:  # gt is valid for viscot
        pass

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "viscot.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[viscot/{condition}] Loaded {len(results)} cached results.")
    else:
        data  = load_viscot(viscot_path, img_root, max_samples)
        start = len(results)
        for idx, item in enumerate(tqdm(data[start:], desc=f"VisCoT/{condition}")):
            gt_latent_embeds = None

            if condition == "gt" and item.get("bbox"):
                gt_latent_embeds = get_gt_latent_embeds(
                    model, processor, item["img_path"], item["bbox"], device)
            elif condition == "random_bbox" and item.get("bbox"):
                next_item = data[(start + idx + 1) % len(data)]
                gt_latent_embeds = get_random_bbox_latent_embeds(
                    model, processor, next_item["img_path"], item["bbox"], device)
            if condition == "random" or condition == "zeros":
                perturbation = condition
            else:
                perturbation = None

            main_img = item["img_path"]
            if corrupt_image and item.get("bbox"):
                from PIL import ImageDraw
                main_img = Image.open(main_img).convert("RGB")
                draw = ImageDraw.Draw(main_img)
                bbox = item["bbox"]
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=(0, 0, 0))

            response  = run_inference(model, processor, main_img,
                                      item["question"], condition, device,
                                      gt_latent_embeds=gt_latent_embeds, perturbation=perturbation)
            pred = extract_mc_answer(response, item["options"])
            gt   = item["label"]
            print(f"     pred={pred!r:3s} gt={gt!r} {'✓' if pred == gt else '✗'}")
            results.append({
                "img_path":    item["img_path"],
                "question":    item["question"],
                "label":       gt,
                "prediction":  response,
                "pred_answer": pred,
            })
            if (idx + 1) % 10 == 0:
                save_cache(out_file, results)
        save_cache(out_file, results)

    correct = sum(r["pred_answer"] == r["label"] for r in results)
    total   = len(results)
    acc     = correct / total if total > 0 else 0.0
    print(f"[viscot/{condition}] {correct}/{total} = {acc:.4f}")
    return {"accuracy": acc, "correct": correct, "total": total}


# ── Blink ──────────────────────────────────────────────────────────────────────

def load_blink(max_samples: Optional[int] = None) -> list:
    data = []
    for cfg in BLINK_CATEGORIES:
        ds = load_dataset("BLINK-Benchmark/BLINK", cfg)["val"]
        for item in ds:
            letters    = string.ascii_uppercase
            options    = [c for c in item["choices"]]
            option_str = "".join(f"{l}. {c}\n" for l, c in zip(letters, options))
            ans = item["answer"][1].upper() if len(item["answer"]) > 1 else item["answer"][0].upper()
            images = [item[k] for k in ["image_1", "image_2", "image_3", "image_4"]
                      if k in item and item[k] is not None]
            data.append({
                "question_id": item["idx"],
                "images":      images,
                "question":    item["question"] + "\nOptions:\n" + option_str,
                "options":     options,
                "label":       ans,
                "category":    cfg,
            })
    return data[:max_samples] if max_samples else data


def evaluate_blink(model, processor, condition: str, out_dir: str, device,
                   max_samples: Optional[int] = None) -> dict:
    if condition == "gt":
        print(f"[blink/{condition}] Skipped — no GT bboxes on Blink.")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "blink.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[blink/{condition}] Loaded {len(results)} cached results.")
    else:
        data  = load_blink(max_samples)
        start = len(results)
        for idx, item in enumerate(tqdm(data[start:], desc=f"Blink/{condition}")):
            gt_latent_embeds = None

            if condition == "zeros" or condition == "random":
                perturbation = condition
            else:
                perturbation = None

            if condition == "random_bbox":
                # use next item's image so the crop never comes from the query image
                next_item = data[(start + idx + 1) % len(data)]
                source_img = next_item["images"][0]
                gt_latent_embeds = get_random_crop_latent_embeds(
                    model, processor, source_img, device)

            response = run_inference(model, processor, item["images"],
                                     item["question"], condition, device,
                                     gt_latent_embeds=gt_latent_embeds, perturbation=perturbation)
            pred = extract_mc_answer(response, item["options"])
            gt   = item["label"]
            print(f"     pred={pred!r:3s} gt={gt!r} {'✓' if pred == gt else '✗'}")
            results.append({
                "id":          item["question_id"],
                "label":       gt,
                "prediction":  response,
                "pred_answer": pred,
                "category":    item["category"],
            })
            if (idx + 1) % 10 == 0:
                save_cache(out_file, results)
        save_cache(out_file, results)

    correct = sum(r["pred_answer"] == r["label"] for r in results)
    total   = len(results)
    acc     = correct / total if total > 0 else 0.0
    print(f"[blink/{condition}] {correct}/{total} = {acc:.4f}")

    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["category"], {"correct": 0, "total": 0})
        by_cat[r["category"]]["total"] += 1
        if r["pred_answer"] == r["label"]:
            by_cat[r["category"]]["correct"] += 1

    return {
        "accuracy": acc, "correct": correct, "total": total,
        "by_category": {cat: c["correct"] / c["total"]
                        for cat, c in by_cat.items() if c["total"] > 0},
    }


# ── VStar ──────────────────────────────────────────────────────────────────────

def evaluate_vstar(model, processor, condition: str, out_dir: str, device,
                   max_samples: Optional[int] = None) -> dict:
    if condition == "gt":
        print(f"[vstar/{condition}] Skipped — no GT bboxes on VStar.")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "vstar.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[vstar/{condition}] Loaded {len(results)} cached results.")
    else:
        ds = load_dataset("lmms-lab/vstar-bench", split="test")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        start = len(results)

        for idx, item in enumerate(tqdm(list(ds)[start:], desc=f"VStar/{condition}")):
            img      = item["image"]
            question = item["text"].split("Answer with the option's letter from the given choices directly.")[0].strip()
            category = item.get("category") or (
                "direct_attributes" if int(item["question_id"]) <= 114
                else "relative_position"
            )

            gt_latent_embeds = None

            if condition == "zeros" or condition == "random":
                perturbation = condition
            else:
                perturbation = None
            if condition == "random_bbox":
                # use next item's image so the crop never comes from the query image
                next_ds_item = ds[(start + idx + 1) % len(ds)]
                source_pil = next_ds_item["image"] if isinstance(next_ds_item["image"], Image.Image) else Image.fromarray(next_ds_item["image"])
                gt_latent_embeds = get_random_crop_latent_embeds(
                    model, processor, source_pil, device)

            response = run_inference(model, processor, img, question,
                                     condition, device, gt_latent_embeds=gt_latent_embeds, perturbation=perturbation)
            pred = extract_mc_answer(response)
            gt   = item["label"]
            print(f"     pred={pred!r:3s} gt={gt!r} {'✓' if pred == gt else '✗'}")
            results.append({
                "id":          item["question_id"],
                "label":       gt,
                "prediction":  response,
                "pred_answer": pred,
                "category":    category,
            })
            if (idx + 1) % 10 == 0:
                save_cache(out_file, results)
        save_cache(out_file, results)

    correct = sum(r["pred_answer"] == r["label"] for r in results)
    total   = len(results)
    acc     = correct / total if total > 0 else 0.0
    print(f"[vstar/{condition}] {correct}/{total} = {acc:.4f}")

    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["category"], {"correct": 0, "total": 0})
        by_cat[r["category"]]["total"] += 1
        if r["pred_answer"] == r["label"]:
            by_cat[r["category"]]["correct"] += 1

    return {
        "accuracy": acc, "correct": correct, "total": total,
        "by_category": {cat: c["correct"] / c["total"]
                        for cat, c in by_cat.items() if c["total"] > 0},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ref",   type=str,
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/")
    parser.add_argument("--latent_size", type=int, default=LATENT_SIZE)
    parser.add_argument("--conditions",  type=str, nargs="+",
        default=["own", "no_lvr", "zeros", "random", "gt", "random_bbox"],
        choices=["own", "no_lvr", "zeros", "random", "gt", "random_bbox", "empty_latent"])
    parser.add_argument("--benchmarks",  type=str, nargs="+",
        default=["viscot", "blink", "vstar"],
        choices=["viscot", "blink", "vstar"])
    parser.add_argument("--output_dir",     type=str, default="results/lantern_ablation")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--viscot_path",    type=str, default=VISCOT_TEST_PATH)
    parser.add_argument("--img_root",       type=str, default=IMG_ROOT)
    parser.add_argument("--corrupt_image",  action="store_true",
        help="Black out bbox region in the main image (mirrors corrupt_bbox_blackout training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_ref}")
    model, processor = load_model(args.model_ref, device_map="cuda",
                                  compute_dtype=torch.bfloat16, use_cache=True)
    set_latent_tokens(processor, model, args.latent_size)
    model.eval()
    print(f"Model loaded. latent_size={args.latent_size}, "
          f"lvr_start_id={model.config.lvr_start_id}")

    print(f"Conditions : {args.conditions}")
    print(f"Benchmarks : {args.benchmarks}\n")

    all_results = {"model": args.model_ref}

    for cond in args.conditions:
        cond_dir     = os.path.join(args.output_dir, cond)
        cond_results = {}

        if "viscot" in args.benchmarks:
            print("=" * 64)
            cond_results["viscot"] = evaluate_viscot(
                model, processor, cond,
                out_dir=os.path.join(cond_dir, "viscot"),
                device=device,
                viscot_path=args.viscot_path,
                img_root=args.img_root,
                max_samples=args.max_samples,
                corrupt_image=args.corrupt_image,
            )

        if "blink" in args.benchmarks:
            print("=" * 64)
            cond_results["blink"] = evaluate_blink(
                model, processor, cond,
                out_dir=os.path.join(cond_dir, "blink"),
                device=device,
                max_samples=args.max_samples,
            )

        if "vstar" in args.benchmarks:
            print("=" * 64)
            cond_results["vstar"] = evaluate_vstar(
                model, processor, cond,
                out_dir=os.path.join(cond_dir, "vstar"),
                device=device,
                max_samples=args.max_samples,
            )

        all_results[cond] = cond_results

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Summary:")
    header = f"{'Condition':<12}" + "".join(f"  {b:>10}" for b in args.benchmarks)
    print(header)
    print("-" * 72)
    for cond in args.conditions:
        row = f"{cond:<12}"
        for bench in args.benchmarks:
            acc = all_results.get(cond, {}).get(bench, {}).get("accuracy", float("nan"))
            row += f"  {acc:>10.4f}"
        print(row)
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
