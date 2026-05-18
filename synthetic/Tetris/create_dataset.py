"""
Dataset creation CLI for the visual analogy task.

Enumerates ALL unique (shape_A, rot_A, rot_step, shape_C, rot_C) configurations,
shuffles them with a fixed seed, splits 95/5 into train/eval (zero overlap),
then generates one sample per config with random grid placement and rendering.

Usage:
    python -m synthetic.Tetris.create_dataset \
        --output_dir /path/to/analogy_dataset \
        --seed 42

Output:
    {output_dir}/images/{sample_id:06d}.png
    {output_dir}/images/intermediate_{sample_id:06d}.png
    {output_dir}/train.json
    {output_dir}/eval.json
"""

import argparse
import json
import os
import random

from tqdm import tqdm

from .analogy_simulator import generate_analogy_sample
from .pieces import HARDER_C_SHAPES, HELD_OUT_SHAPES, SHAPES
from .reasoning import fill_reasoning_traces

# ---------------------------------------------------------------------------
# Question template
# ---------------------------------------------------------------------------

ANALOGY_QUESTION = (
    "Image (A) is to image (B) as image (C) is to which of the following options?\n"
    "The transformation from (A) to (B) is: {transform_description}.\n"
    "Note: the colours of the options are random and irrelevant — answer based on shape only.\n"
    "Options:\n(a) Option a\n(b) Option b\n(c) Option c\n(d) Option d"
)

ANALOGY_QUESTION_DEFAULT_COLORS = (
    "Image (A) is to image (B) as image (C) is to which of the following options?\n"
    "The transformation from (A) to (B) is: {transform_description}.\n"
    "Options:\n(a) Option a\n(b) Option b\n(c) Option c\n(d) Option d"
)


# ---------------------------------------------------------------------------
# Config enumeration
# ---------------------------------------------------------------------------

def enumerate_configs() -> list:
    """
    Return all unique (shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx) tuples.
    shape_C must be from a different family than shape_A.
    """
    configs = []
    for shape_A in SHAPES:
        n_rots_A = len(shape_A["rotations"])
        if n_rots_A < 2:
            continue  # need at least one non-identity rotation
        valid_C = [s for s in SHAPES
                   if s["name"] != shape_A["name"]
                   and s["family"] != shape_A["family"]]
        for rot_A_idx in range(n_rots_A):
            for rot_step in range(1, n_rots_A):
                for shape_C in valid_C:
                    n_rots_C = len(shape_C["rotations"])
                    for rot_C_idx in range(n_rots_C):
                        configs.append((shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx))
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate visual analogy synthetic dataset.")
    p.add_argument("--output_dir", type=str,
                   default="/mnt/scratch-nyx/gviveiros/lantern/analogy_data",
                   help="Directory to save images and JSON.")
    p.add_argument("--grid_size",  type=int, default=8,
                   help="Square grid rows/cols for option panels.")
    p.add_argument("--cell_size",  type=int, default=32,
                   help="Pixel size per grid cell.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--eval_frac",  type=float, default=0.05,
                   help="Fraction of configs reserved exclusively for eval.")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--held_out",        action="store_true",
                   help="Generate held-out OOD test set using HELD_OUT_SHAPES as shape_C. "
                        "Saves to {output_dir}/held_out.json instead of train/eval.")
    p.add_argument("--harder",          action="store_true",
                   help="Generate harder held-out OOD test set: shape_A from HELD_OUT_SHAPES "
                        "(novel hexominoes), shape_C from HARDER_C_SHAPES (novel heptominoes). "
                        "Saves to {output_dir}/held_out_harder/held_out_harder.json.")
    p.add_argument("--held_out_max_samples", type=int, default=500,
                   help="Cap on held-out samples (default: 500).")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap on train samples (default: no cap).")
    p.add_argument("--max_eval_samples",  type=int, default=None,
                   help="Cap on eval samples (default: no cap).")
    p.add_argument("--bw_intermediate", action="store_true", default=True,
                   help="Render intermediate rotation strips in black & white (default: True).")
    p.add_argument("--color_intermediate", action="store_false", dest="bw_intermediate",
                   help="Render intermediate rotation strips in the shape's own color.")
    p.add_argument("--default_option_colors", action="store_true",
                   help="Use each shape's own color for option panels instead of random palette colors.")
    return p.parse_args()


def _generate_record(sample_id, shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx,
                     correct_pos, args, rng, images_dir):
    sample = generate_analogy_sample(
        shape_A=shape_A,
        shape_C=shape_C,
        transform_type="rotation",
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        cell_size=args.cell_size,
        rng=rng,
        force_correct_pos=correct_pos,
        ref_rot_A_idx=rot_A_idx,
        ref_rot_C_idx=rot_C_idx,
        rot_step_override=rot_step,
        bw_intermediate=getattr(args, "bw_intermediate", False),
        randomize_option_colors=not getattr(args, "default_option_colors", False),
    )

    img_filename   = f"{sample_id:06d}.png"
    inter_filename = f"intermediate_{sample_id:06d}.png"
    img_path   = os.path.join(images_dir, img_filename)
    inter_path = os.path.join(images_dir, inter_filename)
    sample["composite_img"].save(img_path)
    sample["intermediate_img"].save(inter_path)

    trace = fill_reasoning_traces(sample, rng=rng)
    trace["intermediate_img_path"] = os.path.join("images", inter_filename)

    return {
        "sample_id":             sample_id,
        "img_path":              os.path.join("images", img_filename),
        "question":              (
            ANALOGY_QUESTION_DEFAULT_COLORS if getattr(args, "default_option_colors", False)
            else ANALOGY_QUESTION
        ).format(transform_description=sample["transform_description"]),
        "answer":                sample["answer"],
        "bboxs":                 sample["bboxes"],
        "reasoning_traces":      trace,
        "dataset":               "tetris_analogy",
        "transform_type":        sample["transform_type"],
        "transform_description": sample["transform_description"],
        "shape_A_name":          sample["shape_A_name"],
        "shape_C_name":          sample["shape_C_name"],
        "shape_A_family":        sample["shape_A_family"],
        "shape_C_family":        sample["shape_C_family"],
        "option_transforms":     sample["option_transforms"],
        # Uniquely identifies the intermediate image: same shape_C + same starting
        # rotation + same rotation step → identical rotation strip.
        "intermediate_key":      f"{sample['shape_C_name']}_{rot_C_idx}_{rot_step}",
    }


def enumerate_harder_held_out_configs() -> list:
    """
    Harder OOD configs:
      shape_A — from HELD_OUT_SHAPES (novel hexominoes, never seen in training)
      shape_C — from HARDER_C_SHAPES (novel heptominoes, never seen in training)

    Harder than the standard held-out set because:
      • shape_C has 7 cells (vs 6) with more complex geometry.
      • The model must apply a transformation observed on an unfamiliar hexomino
        to a visually more complex, also-unfamiliar heptomino.
    """
    configs = []
    for shape_A in HELD_OUT_SHAPES:
        n_rots_A = len(shape_A["rotations"])
        if n_rots_A < 2:
            continue
        for rot_A_idx in range(n_rots_A):
            for rot_step in range(1, n_rots_A):
                for shape_C in HARDER_C_SHAPES:
                    for rot_C_idx in range(len(shape_C["rotations"])):
                        configs.append((shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx))
    return configs


def enumerate_held_out_configs() -> list:
    """
    Fully OOD configs: both shape_A and shape_C are held-out shapes (never seen during training).
    shape_A != shape_C and from different families where possible.
    The model must generalise a rotation seen on one unseen shape and apply it to another.
    """
    configs = []
    for shape_A in HELD_OUT_SHAPES:
        n_rots_A = len(shape_A["rotations"])
        if n_rots_A < 2:
            continue
        valid_C = [s for s in HELD_OUT_SHAPES
                   if s["name"] != shape_A["name"]]
        for rot_A_idx in range(n_rots_A):
            for rot_step in range(1, n_rots_A):
                for shape_C in valid_C:
                    for rot_C_idx in range(len(shape_C["rotations"])):
                        configs.append((shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx))
    return configs


def main():
    args = parse_args()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ── Held-out OOD test set ─────────────────────────────────────────────────
    if args.held_out:
        # Separate subdirectory so held-out images never collide with training images
        held_out_dir        = os.path.join(args.output_dir, "held_out")
        held_out_images_dir = os.path.join(held_out_dir, "images")
        os.makedirs(held_out_images_dir, exist_ok=True)
        held_out_path = os.path.join(held_out_dir, "held_out.json")

        print("Enumerating held-out configurations (OOD shapes as shape_C)...")
        configs = enumerate_held_out_configs()
        rng = random.Random(args.seed)
        rng.shuffle(configs)
        print(f"Total held-out configs: {len(configs)}")

        records = []
        if os.path.exists(held_out_path):
            with open(held_out_path) as f:
                records = json.load(f)
            print(f"Resuming: {len(records)} already done.")
        sample_id = max((r["sample_id"] for r in records), default=-1) + 1

        configs = configs[:args.held_out_max_samples + len(records)]
        held_out_start = len(records)
        for i, (shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx) in enumerate(
                tqdm(configs[len(records):], desc="Held-out", initial=len(records), total=min(len(configs), args.held_out_max_samples))):
            correct_pos = (held_out_start + i) % 4
            try:
                record = _generate_record(sample_id, shape_A, rot_A_idx, rot_step,
                                          shape_C, rot_C_idx, correct_pos, args, rng, held_out_images_dir)
            except Exception as e:
                print(f"  [warn] skipped config {i}: {e}")
                sample_id += 1
                continue
            records.append(record)
            sample_id += 1
            if len(records) % args.save_every == 0:
                with open(held_out_path, "w") as f:
                    json.dump(records, f, indent=2)

        with open(held_out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Held-out done: {len(records)} records → {held_out_path}")
        return

    # ── Harder held-out OOD test set ──────────────────────────────────────────
    if args.harder:
        harder_dir        = os.path.join(args.output_dir, "held_out_harder")
        harder_images_dir = os.path.join(harder_dir, "images")
        os.makedirs(harder_images_dir, exist_ok=True)
        harder_path = os.path.join(harder_dir, "held_out_harder.json")

        print("Enumerating harder held-out configurations (held-out hexomino A, heptomino C)...")
        configs = enumerate_harder_held_out_configs()
        rng = random.Random(args.seed)
        rng.shuffle(configs)
        print(f"Total harder held-out configs: {len(configs)}")

        records = []
        if os.path.exists(harder_path):
            with open(harder_path) as f:
                records = json.load(f)
            print(f"Resuming: {len(records)} already done.")
        sample_id = max((r["sample_id"] for r in records), default=-1) + 1

        configs = configs[:args.held_out_max_samples + len(records)]
        harder_start = len(records)
        for i, (shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx) in enumerate(
                tqdm(configs[len(records):], desc="Harder held-out",
                     initial=len(records), total=min(len(configs), args.held_out_max_samples))):
            correct_pos = (harder_start + i) % 4
            try:
                record = _generate_record(sample_id, shape_A, rot_A_idx, rot_step,
                                          shape_C, rot_C_idx, correct_pos, args, rng, harder_images_dir)
            except Exception as e:
                print(f"  [warn] skipped config {i}: {e}")
                sample_id += 1
                continue
            records.append(record)
            sample_id += 1
            if len(records) % args.save_every == 0:
                with open(harder_path, "w") as f:
                    json.dump(records, f, indent=2)

        with open(harder_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Harder held-out done: {len(records)} records → {harder_path}")
        return

    train_path = os.path.join(args.output_dir, "train.json")
    eval_path  = os.path.join(args.output_dir, "eval.json")

    # ── Enumerate & split configs ─────────────────────────────────────────────
    print("Enumerating configurations...")
    configs = enumerate_configs()
    rng = random.Random(args.seed)
    rng.shuffle(configs)

    n_eval  = int(args.eval_frac * len(configs))
    n_train = len(configs) - n_eval
    eval_configs  = configs[:n_eval]
    train_configs = configs[n_eval:]

    if args.max_eval_samples is not None:
        eval_configs  = eval_configs[:args.max_eval_samples]
    if args.max_train_samples is not None:
        train_configs = train_configs[:args.max_train_samples]
    n_eval  = len(eval_configs)
    n_train = len(train_configs)

    print(f"Total unique configs: {len(configs)}")
    print(f"  Train configs: {n_train}  |  Eval configs: {n_eval}")
    print("  Zero overlap guaranteed between train and eval.")

    # ── Generate eval samples ─────────────────────────────────────────────────
    eval_records = []
    sample_id = 0

    # Check for existing eval checkpoint
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_records = json.load(f)
        sample_id = max(r["sample_id"] for r in eval_records) + 1
        print(f"Resuming eval: {len(eval_records)}/{n_eval} already done.")

    remaining_eval = eval_configs[len(eval_records):]
    eval_start = len(eval_records)
    for i, (shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx) in enumerate(
            tqdm(remaining_eval, desc="Eval samples", initial=eval_start, total=n_eval)):
        correct_pos = (eval_start + i) % 4
        try:
            record = _generate_record(sample_id, shape_A, rot_A_idx, rot_step,
                                      shape_C, rot_C_idx, correct_pos, args, rng, images_dir)
        except Exception as e:
            print(f"  [warn] skipped eval config {i}: {e}")
            sample_id += 1
            continue
        eval_records.append(record)
        sample_id += 1
        if len(eval_records) % args.save_every == 0:
            with open(eval_path, "w") as f:
                json.dump(eval_records, f, indent=2)

    with open(eval_path, "w") as f:
        json.dump(eval_records, f, indent=2)
    print(f"Eval done: {len(eval_records)} records → {eval_path}")

    # ── Generate train samples ────────────────────────────────────────────────
    train_records = []

    if os.path.exists(train_path):
        with open(train_path) as f:
            train_records = json.load(f)
        sample_id = max(r["sample_id"] for r in train_records) + 1
        print(f"Resuming train: {len(train_records)}/{n_train} already done.")

    remaining_train = train_configs[len(train_records):]
    train_start = len(train_records)
    for i, (shape_A, rot_A_idx, rot_step, shape_C, rot_C_idx) in enumerate(
            tqdm(remaining_train, desc="Train samples", initial=train_start, total=n_train)):
        correct_pos = (train_start + i) % 4
        try:
            record = _generate_record(sample_id, shape_A, rot_A_idx, rot_step,
                                      shape_C, rot_C_idx, correct_pos, args, rng, images_dir)
        except Exception as e:
            print(f"  [warn] skipped train config {i}: {e}")
            sample_id += 1
            continue
        train_records.append(record)
        sample_id += 1
        if len(train_records) % args.save_every == 0:
            with open(train_path, "w") as f:
                json.dump(train_records, f, indent=2)

    with open(train_path, "w") as f:
        json.dump(train_records, f, indent=2)
    print(f"Train done: {len(train_records)} records → {train_path}")

    # ── Answer distribution check ─────────────────────────────────────────────
    for split_name, records in [("train", train_records), ("eval", eval_records)]:
        from collections import Counter
        dist = Counter(r["answer"] for r in records)
        total = len(records)
        print(f"{split_name} answer distribution: " +
              ", ".join(f"{k}={v/total:.1%}" for k, v in sorted(dist.items())))


if __name__ == "__main__":
    main()
