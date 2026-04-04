"""
Unified dataset creation CLI for both Tetris problem types.

Usage:
    # Transformation task (reference shape → which option matches the transform?)
    python -m synthetic.Tetris.create_dataset \
        --task tetris \
        --output_dir /path/to/tetris_dataset \
        --n_samples 50000 \
        --seed 42

    # Visual analogy task (A:B :: C:?)
    python -m synthetic.Tetris.create_dataset \
        --task analogy \
        --output_dir /path/to/analogy_dataset \
        --n_samples 50000 \
        --seed 42

Output:
    {output_dir}/images/{sample_id:06d}.png
    {output_dir}/tetris_data.json    (task=tetris)
    {output_dir}/analogy_data.json   (task=analogy)

Reasoning traces are written as empty placeholders (stage 1).
Stage 2 will fill them with interleaved text+image reasoning via a VLM pass.
"""

import argparse
import json
import os
import random
from tqdm import tqdm

from .pieces import SHAPES
from .simulator import generate_sample
from .analogy_simulator import generate_analogy_sample


# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

TETRIS_QUESTION = (
    "The left panel shows a reference shape on a grid. "
    "Which of the following options (a, b, c, d) shows a valid {transform_description} "
    "of the reference shape?\n"
    "Options:\n(a) Option a\n(b) Option b\n(c) Option c\n(d) Option d"
)

ANALOGY_QUESTION = (
    "Image (A) is to image (B) as image (C) is to which of the following options?\n"
    "The transformation from (A) to (B) is: {transform_description}.\n"
    "Options:\n(a) Option a\n(b) Option b\n(c) Option c\n(d) Option d"
)


# ---------------------------------------------------------------------------
# Reasoning trace scaffold
# ---------------------------------------------------------------------------

def _empty_traces(answer: str) -> dict:
    """
    Stage-1 placeholder traces.  Stage 2 will fill these in via a VLM pass.

    Intended stage-2 structure (not yet written to JSON):
        "steps": [
            {"type": "text",    "content": "...pre-inspection reasoning..."},
            {"type": "inspect", "option": "a", "content": "...reasoning about option a..."},
            {"type": "inspect", "option": "b", "content": "..."},
            {"type": "inspect", "option": "c", "content": "..."},
            {"type": "inspect", "option": "d", "content": "..."},
            {"type": "text",    "content": "...final conclusion..."},
        ]
    """
    return {
        "pre_visual_text_think": "",           # text before any bbox inspection
        "post_visual_text_think": ["", "", "", ""],  # one entry per bbox (a/b/c/d)
        "text_think": None,                    # final consolidation text
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Per-task sample generators
# ---------------------------------------------------------------------------

_TETRIS_TRANSFORM_TYPES  = ["rotation", "combined"]   # no translation: same orientation = looks identical to reference
_ANALOGY_TRANSFORM_TYPES = ["rotation", "combined"]
_ELIGIBLE_A = [s for s in SHAPES if len(s["rotations"]) > 1]


def _generate_tetris(sample_id: int, args, rng: random.Random) -> dict:
    shape = rng.choice(_ELIGIBLE_A)
    ttype = rng.choice(_TETRIS_TRANSFORM_TYPES)
    sample = generate_sample(
        shape=shape,
        transform_type=ttype,
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        ref_cell_size=args.cell_size,
        opt_cell_size=max(16, args.cell_size // 2),
        rng=rng,
        force_correct_pos=sample_id % 4,
    )
    return {
        "question": TETRIS_QUESTION.format(
            transform_description=sample["transform_description"]
        ),
        "answer": sample["answer"].lower(),
        "bboxs": sample["bboxes"],
        "reasoning_traces": _empty_traces(sample["answer"].lower()),
        "dataset": "tetris",
        "transform_type": sample["transform_type"],
        "transform_description": sample["transform_description"],
        "shape_name": sample["shape_name"],
        "shape_family": sample["shape_family"],
    }, sample["composite_img"]


def _generate_analogy(sample_id: int, args, rng: random.Random) -> dict:
    shape_A = rng.choice(_ELIGIBLE_A)
    shape_C = rng.choice([
        s for s in SHAPES
        if s["name"] != shape_A["name"] and s["family"] != shape_A["family"]
    ])
    ttype = rng.choice(_ANALOGY_TRANSFORM_TYPES)
    sample = generate_analogy_sample(
        shape_A=shape_A,
        shape_C=shape_C,
        transform_type=ttype,
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        cell_size=args.cell_size,
        rng=rng,
        force_correct_pos=sample_id % 4,
    )
    return {
        "question": ANALOGY_QUESTION.format(
            transform_description=sample["transform_description"]
        ),
        "answer": sample["answer"],
        "bboxs": sample["bboxes"],
        "reasoning_traces": _empty_traces(sample["answer"]),
        "dataset": "tetris_analogy",
        "transform_type": sample["transform_type"],
        "transform_description": sample["transform_description"],
        "shape_A_name": sample["shape_A_name"],
        "shape_C_name": sample["shape_C_name"],
        "shape_A_family": sample["shape_A_family"],
        "shape_C_family": sample["shape_C_family"],
    }, sample["composite_img"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Tetris synthetic datasets.")
    p.add_argument("--task",       type=str, required=True, choices=["tetris", "analogy"],
                   help="Which problem type to generate.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save images and JSON.")
    p.add_argument("--n_samples",  type=int, default=10_000)
    p.add_argument("--grid_size",  type=int, default=8,
                   help="Square grid rows/cols.")
    p.add_argument("--cell_size",  type=int, default=None,
                   help="Pixel size per grid cell (default: 40 for tetris, 32 for analogy).")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--split",      type=str, default="train",
                   choices=["train", "val", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    # Default cell size per task
    if args.cell_size is None:
        args.cell_size = 40 if args.task == "tetris" else 32

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    json_name = "tetris_data.json" if args.task == "tetris" else "analogy_data.json"
    output_json_path = os.path.join(args.output_dir, json_name)

    generator = _generate_tetris if args.task == "tetris" else _generate_analogy

    # Checkpoint recovery
    existing: list = []
    existing_ids: set = set()
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path) as f:
                existing = json.load(f)
            existing_ids = {s["sample_id"] for s in existing if "sample_id" in s}
            print(f"Resuming: {len(existing)} samples already done.")
        except Exception:
            existing = []

    rng = random.Random(args.seed)
    for _ in range(len(existing_ids) * 5):
        rng.random()

    out = list(existing)
    pbar = tqdm(total=args.n_samples, initial=len(out),
                desc=f"Generating {args.task} samples")

    for sample_id in range(args.n_samples):
        if sample_id in existing_ids:
            continue

        try:
            record_fields, img = generator(sample_id, args, rng)
        except Exception as e:
            print(f"\n[WARNING] sample {sample_id} failed: {e}")
            continue

        img_filename = f"{sample_id:06d}.png"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)

        record = {
            "sample_id": sample_id,
            "img_path": os.path.abspath(img_path),
            "split": args.split,
            **record_fields,
        }
        out.append(record)
        pbar.update(1)

        if len(out) % args.save_every == 0:
            with open(output_json_path, "w") as f:
                json.dump(out, f, indent=2)
            pbar.write(f"Checkpoint: {len(out)} samples saved.")

    pbar.close()
    with open(output_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. {len(out)} samples → {output_json_path}")


if __name__ == "__main__":
    main()
