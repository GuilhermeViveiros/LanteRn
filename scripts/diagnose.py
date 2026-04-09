"""
Diagnose train vs eval GT latent utility on 100 samples each.

Usage:
    python scripts/diagnose_train_gt.py \
        --checkpoint /path/to/checkpoint \
        --train_path /path/to/train.json \
        --eval_path  /path/to/eval.json   \
        --n_samples  100

Reports:
    train_gt_acc  — GT latent utility on 100 training samples
    eval_gt_acc   — GT latent utility on 100 eval samples

If train >> eval  → memorisation / text shortcut
If both ~25%      → model not learning the task at all
If both high      → model is genuinely generalising
"""

import argparse
import random
import re
import torch
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.models import load_model
from src.train import set_latent_tokens
from src.datasets.sft_tetris_data import SFTTetrisDataset, collate_fn_generate
from evals import run_batch_inference


# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    m = re.search(r'<answer>\s*([a-d])\s*</answer>', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'\b([a-d])\b', text[::-1], re.IGNORECASE)
    return m.group(1).lower() if m else ""


@torch.no_grad()
def evaluate_gt(model, processor, dataset, n_samples: int, batch_size: int, desc: str):
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset = Subset(dataset, sorted(indices))

    collate = partial(collate_fn_generate, processor=processor)
    loader  = DataLoader(subset, batch_size=batch_size, collate_fn=collate)

    processor.tokenizer.padding_side = "left"

    correct = 0
    total   = 0
    examples = []

    for inputs, labels in tqdm(loader, desc=desc):
        prompt_len = inputs["input_ids"].shape[1]
        out = run_batch_inference(model, inputs, use_lvr=True, use_gt=True)
        seqs = out.sequences if hasattr(out, "sequences") else out
        generated = processor.tokenizer.batch_decode(
            seqs[:, prompt_len:], skip_special_tokens=False
        )
        for gen, gt in zip(generated, labels):
            pred = _extract_answer(gen)
            ok   = int(pred == gt.strip().lower())
            correct += ok
            total   += 1
            if len(examples) < 5:
                examples.append({
                    "gt": gt.strip(), "pred": pred, "ok": ok,
                    "gen_snippet": gen.replace("\n", " "),
                })

    processor.tokenizer.padding_side = "right"
    acc = correct / max(total, 1)
    return acc, examples


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to LantErn checkpoint (local dir)")
    parser.add_argument("--train_path",  default=None,
                        help="train.json path (default: TETRIS_TRAIN_PATH from constants)")
    parser.add_argument("--eval_path",   default=None,
                        help="eval.json path  (default: TETRIS_EVAL_PATH from constants)")
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--latent_size", type=int, default=8)
    args = parser.parse_args()

    # Resolve paths
    if args.train_path is None or args.eval_path is None:
        from src.constants import TETRIS_TRAIN_PATH, TETRIS_EVAL_PATH
        args.train_path = args.train_path or TETRIS_TRAIN_PATH
        args.eval_path  = args.eval_path  or TETRIS_EVAL_PATH

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Train path : {args.train_path}")
    print(f"Eval  path : {args.eval_path}")
    print(f"Samples    : {args.n_samples}  |  batch: {args.batch_size}")

    # Load model
    model, processor = load_model(args.checkpoint, compute_dtype=torch.bfloat16, use_cache=True)
    set_latent_tokens(processor, model, args.latent_size)
    model.eval()
    model = model.cuda()

    # Load datasets
    train_ds = SFTTetrisDataset(args.train_path, processor, use_lvr=True)
    eval_ds  = SFTTetrisDataset(args.eval_path,  processor, use_lvr=True)

    print(f"\nDataset sizes — train: {len(train_ds)}  eval: {len(eval_ds)}")

    # Evaluate
    train_acc, train_ex = evaluate_gt(model, processor, train_ds,
                                      args.n_samples, args.batch_size, "train GT")
    eval_acc,  eval_ex  = evaluate_gt(model, processor, eval_ds,
                                      args.n_samples, args.batch_size, "eval  GT")

    # Report
    print("\n" + "=" * 55)
    print(f"  train GT accuracy : {train_acc:.1%}  ({args.n_samples} samples)")
    print(f"  eval  GT accuracy : {eval_acc:.1%}  ({args.n_samples} samples)")
    print("=" * 55)

    gap = train_acc - eval_acc
    if gap > 0.3:
        verdict = "MEMORISATION / TEXT SHORTCUT  (train >> eval)"
    elif train_acc < 0.35 and eval_acc < 0.35:
        verdict = "NOT LEARNING  (both near chance)"
    else:
        verdict = "GENERALISING  (train ≈ eval, both above chance)"
    print(f"  Verdict: {verdict}")
    print("=" * 55)

    print("\n--- Train examples (first 5) ---")
    for i, ex in enumerate(train_ex):
        mark = "✓" if ex["ok"] else "✗"
        print(f"  [{mark}] gt={ex['gt']}  pred={ex['pred']}  | {ex['gen_snippet']}")

    print("\n--- Eval examples (first 5) ---")
    for i, ex in enumerate(eval_ex):
        mark = "✓" if ex["ok"] else "✗"
        print(f"  [{mark}] gt={ex['gt']}  pred={ex['pred']}  | {ex['gen_snippet']}")


if __name__ == "__main__":
    main()


# python scripts/diagnose.py --checkpoint /e/project1/jureap131/gviveiros/lantern/checkpoints/lantern_tetris_sft_lt_8_lambda_0.1/checkpoint-800 --n_samples 100 --batch_size 8
