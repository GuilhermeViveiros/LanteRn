"""
OOD generalization eval: compare GT vs own latent accuracy on held-out shapes
(shapes never seen during training) across multiple checkpoints.

Held-out shapes: H_hook, H_F, H_G, H_Y, H_R  (hexominoes not in training data)

Generate held-out data first:
    python -m synthetic.Tetris.create_dataset \
        --output_dir /path/to/analogy_data \
        --held_out

    Images saved to: {output_dir}/held_out/images/
    JSON saved to:   {output_dir}/held_out/held_out.json

Evaluate LVR model (GT + own latents):
    python -m evals.generalization \
        --checkpoints_dir /path/to/checkpoints \
        --held_out_path   /path/to/analogy_data/held_out/held_out.json \
        --eval_path       /path/to/analogy_data/eval.json \
        --n_samples 200   --batch_size 8

Evaluate NTP model (no latent tokens):
    python -m evals.generalization \
        --checkpoints_dir /path/to/ntp_checkpoints \
        --held_out_path   /path/to/analogy_data/held_out/held_out.json \
        --eval_path       /path/to/analogy_data/eval.json \
        --no_lvr
"""

import argparse
import glob
import os
import random
import re
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from evals import run_batch_inference
from src.datasets.sft_tetris_data import (
    SFTTetrisDataset,
    collate_fn_generate,
    collate_fn_generate_ntp,
)
from src.models import load_model
from src.train import set_latent_tokens


def build_distractor_map(dataset: SFTTetrisDataset, seed: int = 42) -> dict:
    """
    For each sample index, return the index of a different sample (any shape,
    any transform) to use as a distractor intermediate image.
    Guarantees no identity mapping.
    """
    rng = random.Random(seed)
    n = len(dataset)
    indices = list(range(n))
    shuffled = indices.copy()
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(shuffled[i] != i for i in range(n)):
            break
    return dict(zip(indices, shuffled))


def build_same_shape_diff_rotation_map(dataset: SFTTetrisDataset, seed: int = 42) -> dict:
    """
    For each sample i, find sample j with same shape_C_name but different transform_description.
    Tests whether a wrong-rotation latent from the same shape confuses the model.
    Accuracy is still measured against i's original correct answer.
    """
    rng = random.Random(seed)
    by_shape = {}
    for i, s in enumerate(dataset.dataset):
        by_shape.setdefault(s["shape_C_name"], []).append(i)

    distractor_map = {}
    for i, s in enumerate(dataset.dataset):
        shape     = s["shape_C_name"]
        transform = s["transform_description"]
        candidates = [j for j in by_shape.get(shape, [])
                      if j != i and dataset.dataset[j]["transform_description"] != transform]
        if candidates:
            distractor_map[i] = rng.choice(candidates)
    return distractor_map


def build_latent_consistent_map(dataset: SFTTetrisDataset, seed: int = 42) -> tuple[dict, dict]:
    """
    For each sample index i, find another sample j such that:
      - j has the same shape_C_name as i
      - j's transform_description appears in i's option_transforms (as a non-correct option)
    Returns:
      distractor_map:  {i → j}  (inject j's intermediate image into i's question)
      expected_answer: {i → option_letter}  (the option in i consistent with j's rotation)
    """
    rng = random.Random(seed)
    # Group by (shape_C_name, transform_description)
    by_shape_transform = {}
    for i, s in enumerate(dataset.dataset):
        key = (s["shape_C_name"], s["transform_description"])
        by_shape_transform.setdefault(key, []).append(i)

    distractor_map = {}
    expected_answer = {}
    for i, s in enumerate(dataset.dataset):
        shape       = s["shape_C_name"]
        correct_t   = s["transform_description"]
        opt_transforms = s.get("option_transforms", {})

        # Find transforms that are (a) in i's options, (b) not the correct one
        candidate_transforms = [t for t in opt_transforms.values()
                                 if t != correct_t and t not in ("perturbed", "other_shape")]
        if not candidate_transforms:
            continue

        # Pick a random target transform T_j and find a sample j with same shape + T_j
        rng.shuffle(candidate_transforms)
        for t_j in candidate_transforms:
            pool = [j for j in by_shape_transform.get((shape, t_j), []) if j != i]
            if pool:
                j = rng.choice(pool)
                distractor_map[i] = j
                # The latent-consistent answer is the option letter in i that shows T_j
                expected_answer[i] = next(k for k, v in opt_transforms.items() if v == t_j)
                break

    return distractor_map, expected_answer


class DistractorDataset(Dataset):
    """Wraps SFTTetrisDataset; replaces each sample's intermediate image with
    one from a different sample to ablate latent content."""

    def __init__(self, base: SFTTetrisDataset, distractor_map: dict):
        self.base = base
        self.distractor_map = distractor_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]                         # [..., latent_visual_dict]
        d_idx  = self.distractor_map[idx]
        distractor_sample = self.base[d_idx]
        sample[-1] = distractor_sample[-1]              # swap latent visual only
        return sample


def _extract_answer(text: str) -> str:
    m = re.search(r'<answer>\s*([a-d])\s*</answer>', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'\b([a-d])\b', text[::-1], re.IGNORECASE)
    return m.group(1).lower() if m else ""


@torch.no_grad()
def evaluate(model, processor, dataset, n_samples: int, batch_size: int,
             use_gt: bool, use_lvr: bool, desc: str):
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset  = Subset(dataset, sorted(indices))

    if use_lvr:
        collate = partial(collate_fn_generate, processor=processor)
    else:
        collate = partial(collate_fn_generate_ntp, processor=processor)
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate)

    processor.tokenizer.padding_side = "left"
    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        prompt_len = inputs["input_ids"].shape[1]
        out  = run_batch_inference(model, inputs, use_lvr=use_lvr, use_gt=use_gt)
        seqs = out.sequences if hasattr(out, "sequences") else out
        generated = processor.tokenizer.batch_decode(
            seqs[:, prompt_len:], skip_special_tokens=False
        )
        for gen, gt in zip(generated, labels):
            correct += int(_extract_answer(gen) == gt.strip().lower())
            total   += 1
    processor.tokenizer.padding_side = "right"
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_latent_consistent(model, processor, dataset, expected_answer: dict,
                                n_samples: int, batch_size: int, desc: str = "latent-consistent"):
    """
    Run inference on DistractorDataset and measure whether the model picks
    the option that is consistent with the injected distractor's rotation,
    as defined by expected_answer[i].
    Only evaluates samples that have a valid expected_answer entry.
    """
    valid_indices = sorted(expected_answer.keys())
    rng = random.Random(42)
    sampled = rng.sample(valid_indices, min(n_samples, len(valid_indices)))
    subset  = Subset(dataset, sorted(sampled))
    # Build a lookup: position in subset → expected letter
    sorted_sampled = sorted(sampled)
    exp_list = [expected_answer[i] for i in sorted_sampled]

    collate = partial(collate_fn_generate, processor=processor)
    loader  = DataLoader(subset, batch_size=batch_size, collate_fn=collate)

    processor.tokenizer.padding_side = "left"
    correct, total = 0, 0
    pos = 0
    for inputs, _ in tqdm(loader, desc=desc, leave=False):
        prompt_len = inputs["input_ids"].shape[1]
        out  = run_batch_inference(model, inputs, use_lvr=True, use_gt=True)
        seqs = out.sequences if hasattr(out, "sequences") else out
        generated = processor.tokenizer.batch_decode(
            seqs[:, prompt_len:], skip_special_tokens=False
        )
        for gen in generated:
            expected = exp_list[pos]
            correct += int(_extract_answer(gen) == expected.lower())
            total   += 1
            pos     += 1
    processor.tokenizer.padding_side = "right"
    return correct / max(total, 1)


def load_and_eval(checkpoint: str, args, eval_ds, held_out_ds,
                  distractor_eval_ds, distractor_held_ds,
                  ss_distractor_eval_ds, ss_distractor_held_ds,
                  lc_eval_ds, lc_eval_expected_answer,
                  lc_held_ds, lc_expected_answer):
    print(f"\n── {os.path.basename(checkpoint)} ──")
    model, processor = load_model(checkpoint, compute_dtype=torch.bfloat16, use_cache=True,
                                  attn_implementation="flash_attention_2")
    set_latent_tokens(processor, model, args.latent_size)
    model.eval().cuda()

    if args.use_lvr:
        eval_acc      = evaluate(model, processor, eval_ds,            args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="eval GT")
        eval_own      = evaluate(model, processor, eval_ds,            args.n_samples, args.batch_size,
                                 use_gt=False, use_lvr=True, desc="eval own")
        eval_distract = evaluate(model, processor, distractor_eval_ds, args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="eval distractor")
        eval_ss       = evaluate(model, processor, ss_distractor_eval_ds, args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="eval same-shape")
        eval_lc       = evaluate_latent_consistent(model, processor, lc_eval_ds,
                                                   lc_eval_expected_answer,
                                                   args.n_samples, args.batch_size,
                                                   desc="eval latent-consistent")
        held_gt       = evaluate(model, processor, held_out_ds,        args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="held GT")
        held_own      = evaluate(model, processor, held_out_ds,        args.n_samples, args.batch_size,
                                 use_gt=False, use_lvr=True, desc="held own")
        held_distract = evaluate(model, processor, distractor_held_ds, args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="held distractor")
        held_ss       = evaluate(model, processor, ss_distractor_held_ds, args.n_samples, args.batch_size,
                                 use_gt=True,  use_lvr=True, desc="held same-shape")
        held_lc       = evaluate_latent_consistent(model, processor, lc_held_ds,
                                                   lc_expected_answer,
                                                   args.n_samples, args.batch_size,
                                                   desc="held latent-consistent")
        result = (os.path.basename(checkpoint),
                  eval_acc, eval_own, eval_distract, eval_ss, eval_lc,
                  held_gt, held_own, held_distract, held_ss, held_lc)
    else:
        eval_acc = evaluate(model, processor, eval_ds,     args.n_samples, args.batch_size,
                            use_gt=False, use_lvr=False, desc="eval NTP")
        held_acc = evaluate(model, processor, held_out_ds, args.n_samples, args.batch_size,
                            use_gt=False, use_lvr=False, desc="held NTP")
        result = (os.path.basename(checkpoint), eval_acc, None, None, None, None, held_acc, None, None, None, None)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", required=True,
                        help="Directory containing checkpoint-* subdirs")
    parser.add_argument("--held_out_path",   required=True,
                        help="held_out.json generated with --held_out flag")
    parser.add_argument("--eval_path",       default=None,
                        help="eval.json for in-distribution comparison")
    parser.add_argument("--n_samples",   type=int, default=200)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--no_lvr", action="store_true",
                        help="Evaluate NTP model (no latent visual tokens)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert intermediate images to grayscale (for gray-trained checkpoints)")
    args = parser.parse_args()
    args.use_lvr = not args.no_lvr

    if args.eval_path is None:
        from src.constants import TETRIS_EVAL_PATH
        args.eval_path = TETRIS_EVAL_PATH

    # Find checkpoints sorted by step number.
    # Accepts either a parent dir containing checkpoint-* subdirs,
    # or a direct path to a single checkpoint-* dir.
    p = args.checkpoints_dir.rstrip("/")
    if os.path.basename(p).startswith("checkpoint-"):
        ckpts = [p]
    else:
        ckpts = sorted(
            glob.glob(os.path.join(p, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1])
        )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoints_dir}")
    print(f"Found {len(ckpts)} checkpoints: {[os.path.basename(c) for c in ckpts]}")
    print(f"Held-out path : {args.held_out_path}")
    print(f"Eval path     : {args.eval_path}")
    print(f"Mode          : {'LVR (GT + own)' if args.use_lvr else 'NTP'}")

    # Load datasets once
    dummy_model, processor = load_model(ckpts[0], compute_dtype=torch.bfloat16, use_cache=False,
                                        attn_implementation="flash_attention_2")
    set_latent_tokens(processor, dummy_model, args.latent_size)
    del dummy_model
    torch.cuda.empty_cache()

    eval_ds     = SFTTetrisDataset(args.eval_path,     processor, use_lvr=args.use_lvr,
                                   grayscale_intermediate=args.grayscale)
    held_out_ds = SFTTetrisDataset(args.held_out_path, processor, use_lvr=args.use_lvr,
                                   grayscale_intermediate=args.grayscale)
    print(f"Eval samples    : {len(eval_ds)}")
    print(f"Held-out samples: {len(held_out_ds)}  "
          f"(shapes: {sorted({s['shape_C_name'] for s in held_out_ds.dataset})})")

    # Build distractor datasets
    distractor_eval_ds = distractor_held_ds = None
    ss_distractor_eval_ds = ss_distractor_held_ds = None
    lc_eval_ds = lc_eval_expected_answer = None
    lc_held_ds = lc_expected_answer = None
    if args.use_lvr:
        eval_distractor_map  = build_distractor_map(eval_ds)
        distractor_eval_ds   = DistractorDataset(eval_ds, eval_distractor_map)
        print(f"Eval distractor : {len(eval_distractor_map)} entries (random intermediate, seed=42)")

        ss_eval_map = build_same_shape_diff_rotation_map(eval_ds)
        ss_distractor_eval_ds = DistractorDataset(eval_ds, ss_eval_map)
        print(f"Eval same-shape : {len(ss_eval_map)} entries (same shape, diff rotation, seed=42)")

        lc_eval_map, lc_eval_expected_answer = build_latent_consistent_map(eval_ds)
        lc_eval_ds = DistractorDataset(eval_ds, lc_eval_map)
        print(f"Eval LC         : {len(lc_eval_map)} valid pairs (same shape_C, different rotation in options, seed=42)")

        held_distractor_map  = build_distractor_map(held_out_ds)
        distractor_held_ds   = DistractorDataset(held_out_ds, held_distractor_map)
        print(f"Held distractor : {len(held_distractor_map)} entries (random intermediate, seed=42)")

        ss_held_map = build_same_shape_diff_rotation_map(held_out_ds)
        ss_distractor_held_ds = DistractorDataset(held_out_ds, ss_held_map)
        print(f"Held same-shape : {len(ss_held_map)} entries (same shape, diff rotation, seed=42)")

        lc_map, lc_expected_answer = build_latent_consistent_map(held_out_ds)
        lc_held_ds = DistractorDataset(held_out_ds, lc_map)
        print(f"Held LC         : {len(lc_map)} valid pairs "
              f"(same shape_C, different rotation in options, seed=42)")

    # Evaluate each checkpoint
    results = []
    for ckpt in ckpts:
        results.append(load_and_eval(ckpt, args, eval_ds, held_out_ds,
                                     distractor_eval_ds, distractor_held_ds,
                                     ss_distractor_eval_ds, ss_distractor_held_ds,
                                     lc_eval_ds, lc_eval_expected_answer,
                                     lc_held_ds, lc_expected_answer))

    # Print table
    print("\n" + "=" * 140)
    if args.use_lvr:
        print(f"{'checkpoint':<25} {'eval_gt':>8} {'eval_own':>9} {'eval_dist':>10} {'eval_ss':>8} {'eval_lc':>8} "
              f"{'held_gt':>8} {'held_own':>9} {'held_dist':>10} {'held_ss':>8} {'held_lc':>8}")
        print("-" * 140)
        for name, eval_acc, eval_own, eval_dist, eval_ss, eval_lc, held_gt, held_own, held_dist, held_ss, held_lc in results:
            print(f"{name:<25} {eval_acc:>7.1%} {eval_own:>9.1%} {eval_dist:>10.1%} {eval_ss:>8.1%} {eval_lc:>8.1%} "
                  f"{held_gt:>8.1%} {held_own:>9.1%} {held_dist:>10.1%} {held_ss:>8.1%} {held_lc:>8.1%}")
        print("=" * 140)
        print("eval_own   = in-distribution accuracy with model's own latents")
        print("eval_dist  = in-distribution accuracy with random distractor latent (any shape/rotation)")
        print("eval_ss    = in-distribution accuracy with same-shape different-rotation distractor")
        print("eval_lc    = in-distribution latent-consistent accuracy (chance=25%)")
        print("held_dist  = OOD accuracy with random distractor latent")
        print("held_ss    = OOD accuracy with same-shape different-rotation distractor")
        print("held_lc    = OOD latent-consistent accuracy (chance=25%)")
    else:
        print(f"{'checkpoint':<25} {'eval':>8} {'held_out':>10}")
        print("-" * 45)
        for name, eval_acc, _, _, _, _, held_acc, _, _, _, _ in results:
            print(f"{name:<25} {eval_acc:>7.1%} {held_acc:>10.1%}")
        print("=" * 45)


if __name__ == "__main__":
    main()
