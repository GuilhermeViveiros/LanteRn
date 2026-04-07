"""
Baseline Qwen2.5-VL evaluation on VisCoT / Blink / VStar.

Runs the stock Qwen2.5-VL model (no LantErn, no LVR) so results can be
compared directly against viscot_blink_vstar_eval.py ablation conditions.

Usage:
  python -m evals.baseline_qwen_eval \\
      --model_ref Qwen/Qwen2.5-VL-7B-Instruct \\
      --benchmarks viscot blink vstar \\
      --output_dir results/baseline_qwen

  # Sanity check
  python -m evals.baseline_qwen_eval \\
      --model_ref Qwen/Qwen2.5-VL-3B-Instruct \\
      --benchmarks viscot --max_samples 5 \\
      --output_dir results/baseline_debug
"""

import os
import json
import string
import argparse
from typing import Optional
import re
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.utils import extract_mc_answer

VISCOT_TEST_PATH = "/e/project1/jureap126/gviveiros/lantern/viscot_mc_test.jsonl"
IMG_ROOT         = "/e/project1/jureap126/gviveiros/lantern/"
BLINK_CATEGORIES = ["Object_Localization", "Spatial_Relation"]


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


# ── Core inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, processor, images, question: str,
                  device, max_new_tokens: int = 512) -> str:
    if not isinstance(images, list):
        images = [images]

    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": (Image.open(img).convert("RGB") if isinstance(img, str) else img)}
          for img in images],
        {"type": "text", "text": question},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True,
                       return_tensors="pt").to(device)

    prompt_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    generated = output_ids[0, prompt_len:]
    decoded   = processor.decode(generated, skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
    print(f"  >> toks={generated.shape[0]} | {repr(decoded)}")
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
            img_path = s.get("img_path", s.get("image", "")).replace(
                "/mnt/data-artemis/gviveiros/lantern/", img_root
            )
            options_clean = [o.split(" ", 1)[1] if " " in o else o for o in options]
            question = s["question"] + "\nOptions:\n" + "\n".join(options)
            data.append({
                "img_path": img_path,
                "question": question,
                "options":  options_clean,
                "label":    s.get("answer", ""),
            })
    return data[:max_samples] if max_samples else data


def evaluate_viscot(model, processor, out_dir: str, device,
                    viscot_path: str = VISCOT_TEST_PATH, img_root: str = IMG_ROOT,
                    max_samples: Optional[int] = None, max_new_tokens: int = 512) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "viscot.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[viscot/baseline] Loaded {len(results)} cached results.")
    else:
        data  = load_viscot(viscot_path, img_root, max_samples)
        start = len(results)
        for idx, item in enumerate(tqdm(data[start:], desc="VisCoT/baseline")):
            response = run_inference(model, processor, item["img_path"],
                                     item["question"], device, max_new_tokens)
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
    print(f"[viscot/baseline] {correct}/{total} = {acc:.4f}")
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


def evaluate_blink(model, processor, out_dir: str, device,
                   max_samples: Optional[int] = None, max_new_tokens: int = 512) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "blink.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[blink/baseline] Loaded {len(results)} cached results.")
    else:
        data  = load_blink(max_samples)
        start = len(results)
        for idx, item in enumerate(tqdm(data[start:], desc="Blink/baseline")):
            response = run_inference(model, processor, item["images"],
                                     item["question"], device, max_new_tokens)
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
    print(f"[blink/baseline] {correct}/{total} = {acc:.4f}")

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

def evaluate_vstar(model, processor, out_dir: str, device,
                   max_samples: Optional[int] = None, max_new_tokens: int = 512) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "vstar.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[vstar/baseline] Loaded {len(results)} cached results.")
    else:
        ds = load_dataset("lmms-lab/vstar-bench", split="test")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        start = len(results)

        for idx, item in enumerate(tqdm(list(ds)[start:], desc="VStar/baseline")):
            img      = item["image"]
            question = item["text"]
            category = item.get("category") or (
                "direct_attributes" if int(item["question_id"]) <= 114
                else "relative_position"
            )

            response = run_inference(model, processor, img, question,
                                     device, max_new_tokens)
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
    print(f"[vstar/baseline] {correct}/{total} = {acc:.4f}")

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


# ── Tetris ────────────────────────────────────────────────────────────────────

TETRIS_DATA_PATH = "/e/project1/jureap131/gviveiros/lantern/analogy_data/eval.json"

def load_tetris_eval(data_path: str, max_samples=None) -> list:
    """Load the pre-split eval set generated by create_dataset.py."""
    with open(data_path) as f:
        samples = json.load(f)
    return samples[:max_samples] if max_samples else samples


def evaluate_tetris(model, processor, out_dir: str, device,
                    data_path: str = TETRIS_DATA_PATH,
                    max_samples=None, max_new_tokens: int = 128) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "tetris.json")
    results  = load_cache(out_file)

    if _cache_complete(out_file, max_samples):
        print(f"[tetris/baseline] Loaded {len(results)} cached results.")
    else:
        data  = load_tetris_eval(data_path, max_samples)
        start = len(results)
        fmt = "\nThink step by step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags with only the option letter (a/b/c/d)."
        for idx, item in enumerate(tqdm(data[start:], desc="Tetris/baseline")):
            question = item["question"] + fmt
            response = run_inference(model, processor, item["img_path"],
                                     question, device, max_new_tokens)
            # prefer <answer> tag, fall back to general extraction
            m = re.search(r'<answer>\s*([a-dA-D])\s*</answer>', response)
            pred = (m.group(1) if m else extract_mc_answer(response) or "").upper()
            gt   = item["answer"].upper()
            print(f"     full: {response!r}")
            print(f"     pred={pred!r:3s} gt={gt!r} {'✓' if pred == gt else '✗'}")
            results.append({
                "sample_id":      item["sample_id"],
                "img_path":       item["img_path"],
                "label":          gt,
                "prediction":     response,
                "pred_answer":    pred,
                "transform_type": item.get("transform_type", ""),
            })
            if (idx + 1) % 20 == 0:
                save_cache(out_file, results)
        save_cache(out_file, results)

    correct = sum(r["pred_answer"] == r["label"] for r in results)
    total   = len(results)
    acc     = correct / total if total > 0 else 0.0
    print(f"[tetris/baseline] {correct}/{total} = {acc:.4f}")

    by_transform: dict = {}
    for r in results:
        t = r.get("transform_type", "unknown")
        by_transform.setdefault(t, {"correct": 0, "total": 0})
        by_transform[t]["total"] += 1
        if r["pred_answer"] == r["label"]:
            by_transform[t]["correct"] += 1

    return {
        "accuracy": acc, "correct": correct, "total": total,
        "by_transform": {t: v["correct"] / v["total"]
                         for t, v in by_transform.items() if v["total"] > 0},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ref",      type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID or local path for the base Qwen2.5-VL model")
    parser.add_argument("--benchmarks",     type=str, nargs="+",
        default=["viscot", "blink", "vstar"],
        choices=["viscot", "blink", "vstar", "tetris"])
    parser.add_argument("--output_dir",     type=str, default="results/baseline_qwen")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--viscot_path",    type=str, default=VISCOT_TEST_PATH)
    parser.add_argument("--img_root",       type=str, default=IMG_ROOT)
    parser.add_argument("--tetris_data_path", type=str, default=TETRIS_DATA_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {args.model_ref}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_ref,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(args.model_ref)
    model.eval()
    print("Model loaded.")

    print(f"Benchmarks: {args.benchmarks}\n")
    results = {"model": args.model_ref}

    if "viscot" in args.benchmarks:
        print("=" * 64)
        results["viscot"] = evaluate_viscot(
            model, processor,
            out_dir=os.path.join(args.output_dir, "viscot"),
            device=device,
            viscot_path=args.viscot_path,
            img_root=args.img_root,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

    if "blink" in args.benchmarks:
        print("=" * 64)
        results["blink"] = evaluate_blink(
            model, processor,
            out_dir=os.path.join(args.output_dir, "blink"),
            device=device,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

    if "tetris" in args.benchmarks:
        print("=" * 64)
        results["tetris"] = evaluate_tetris(
            model, processor,
            out_dir=os.path.join(args.output_dir, "tetris"),
            device=device,
            data_path=args.tetris_data_path,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

    if "vstar" in args.benchmarks:
        print("=" * 64)
        results["vstar"] = evaluate_vstar(
            model, processor,
            out_dir=os.path.join(args.output_dir, "vstar"),
            device=device,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print("Summary  (baseline Qwen2.5-VL):")
    print("-" * 64)
    for bench in args.benchmarks:
        acc = results.get(bench, {}).get("accuracy", float("nan"))
        print(f"  {bench:<10}  {acc:.4f}")
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()

# python -m evals.baseline_qwen_eval --model_ref Qwen/Qwen2.5-VL-7B-Instruct --output_dir results/baseline_qwen
# python -m evals.baseline_qwen_eval --model_ref Qwen/Qwen2.5-VL-3B-Instruct --benchmarks tetris --output_dir baseline_tetris_2b